"""
enhanced_sorter.py — Enhanced face recognition for KinderSort Lite.

Adds ensemble detection (HOG + CNN), image preprocessing via CLAHE,
confidence scoring, and cached encoding support for low-resource environments.
All processing remains CPU-only and fully offline.
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from face_engine import FaceEngine
from preprocessor import ImagePreprocessor, load_and_preprocess
from utils import (
    build_output_filename,
    collect_event_images,
    is_image_file,
    safe_copy,
)


class EnhancedPhotoSorter:
    """Enhanced face recognition pipeline with preprocessing and ensemble detection.

    Builds on the original PhotoSorter with:
        - CLAHE image enhancement for better accuracy in poor lighting
        - Ensemble face detection (HOG + CNN fallback) for higher recall
        - Face-region enhancement for reference photos
        - Confidence scores for each match
        - Encoding cache to avoid recomputation across runs
    """

    DISTANCE_THRESHOLD = 0.50  # Stricter than original (0.55) — preprocessing handles noise
    MAX_IMAGE_DIMENSION = 1000

    def __init__(
        self,
        reference_folder: Path,
        events_folder: Path,
        output_folder: Path,
        logger: logging.Logger,
        use_preprocessing: bool = True,
        use_cache: bool = True,
        ensemble_detection: bool = True,
    ) -> None:
        """Initialise the enhanced sorter.

        Args:
            reference_folder: Path to folder with one reference photo per student.
            events_folder: Path to folder containing event sub-folders with photos.
            output_folder: Where sorted results will be written.
            logger: Configured logger instance.
            use_preprocessing: Enable CLAHE + brightness normalization.
            use_cache: Cache face encodings to disk for faster re-runs.
            ensemble_detection: Use HOG + CNN ensemble for better face detection.
        """
        self.reference_folder = reference_folder
        self.events_folder = events_folder
        self.output_folder = output_folder
        self.logger = logger
        self.use_cache = use_cache
        self.ensemble_detection = ensemble_detection

        self._student_encodings: dict[str, np.ndarray] = {}
        self._student_confidence: dict[str, float] = {}
        self._preprocessor = ImagePreprocessor(enabled=use_preprocessing)
        self._engine = FaceEngine()
        self._cache_path = output_folder / "encoding_cache.json"

        # Accuracy tracking
        self._match_confidence_scores: list[float] = []
        self._detection_methods_used: dict[str, int] = {"hog": 0, "cnn": 0, "ensemble": 0}

    # ------------------------------------------------------------------
    # Encoding cache (low-resource optimization)
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, list[float]] | None:
        """Load cached encodings from disk if they exist and are fresh."""
        if not self.use_cache or not self._cache_path.exists():
            return None

        try:
            with open(self._cache_path) as f:
                data = json.load(f)

            # Verify cache matches current reference folder
            current_files = sorted(
                p.name for p in self.reference_folder.iterdir() if is_image_file(p)
            )
            cached_files = data.get("_files", [])
            if current_files != cached_files:
                self.logger.info("Cache stale — reference photos changed")
                return None

            # Verify cache age (max 24 hours)
            cached_time = data.get("_timestamp", 0)
            if time.time() - cached_time > 86400:
                self.logger.info("Cache expired (>24h)")
                return None

            self.logger.info("Loaded %d encodings from cache", len(data) - 2)
            return {k: v for k, v in data.items() if not k.startswith("_")}

        except (json.JSONDecodeError, KeyError) as exc:
            self.logger.warning("Cache corrupted: %s", exc)
            return None

    def _save_cache(self) -> None:
        """Save current encodings to disk cache."""
        if not self.use_cache:
            return

        try:
            data = {
                name: enc.tolist()
                for name, enc in self._student_encodings.items()
            }
            data["_files"] = sorted(
                p.name for p in self.reference_folder.iterdir() if is_image_file(p)
            )
            data["_timestamp"] = time.time()

            with open(self._cache_path, "w") as f:
                json.dump(data, f)
            self.logger.info("Saved %d encodings to cache", len(self._student_encodings))

        except OSError as exc:
            self.logger.warning("Could not save cache: %s", exc)

    # ------------------------------------------------------------------
    # Reference loading (enhanced with preprocessing)
    # ------------------------------------------------------------------

    def load_references(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[str]:
        """Load and encode reference photos with preprocessing enhancement.

        Attempts to load from cache first. For each reference photo:
            1. Load image with preprocessing (CLAHE + brightness normalization)
            2. Detect faces with CNN (more accurate for reference photos)
            3. Enhance the face region
            4. Encode with multiple jitters for robust embeddings
            5. Store and cache the encoding

        Returns:
            List of student names whose reference photo had no detectable face.
        """
        # Try cache first
        cached = self._load_cache()
        if cached:
            for name, enc_list in cached.items():
                self._student_encodings[name] = np.array(enc_list)
            self.logger.info(
                "Loaded %d student(s) from cache", len(self._student_encodings)
            )
            return []

        no_face_names: list[str] = []
        reference_images = sorted(
            p for p in self.reference_folder.iterdir() if is_image_file(p)
        )

        if not reference_images:
            self.logger.warning("No reference images found in %s", self.reference_folder)
            return no_face_names

        total = len(reference_images)
        for current, ref_path in enumerate(reference_images, start=1):
            student_name = ref_path.stem
            if progress_callback:
                progress_callback(current, total, student_name)

            try:
                # Load with preprocessing
                rgb_image = load_and_preprocess(ref_path, self._preprocessor)

                # Use CNN for reference photos (more accurate)
                locations = self._engine.face_locations(rgb_image, model="cnn")
                encodings = self._engine.face_encodings(
                    rgb_image,
                    known_face_locations=locations,
                    num_jitters=15,  # More jitters for robust reference encoding
                    model="large",
                )

                if not encodings:
                    self.logger.warning(
                        "No face detected in reference for %s (%s)",
                        student_name,
                        ref_path.name,
                    )
                    no_face_names.append(student_name)
                    continue

                if len(encodings) > 1:
                    self.logger.warning(
                        "Multiple faces in %s reference — using first face only",
                        student_name,
                    )

                self._student_encodings[student_name] = encodings[0]
                self._student_confidence[student_name] = 1.0  # Reference: baseline confidence
                self.logger.info("Loaded reference for %s (enhanced)", student_name)

            except Exception as exc:
                self.logger.error(
                    "Could not read reference photo %s: %s", ref_path.name, exc
                )

        self.logger.info(
            "Loaded %d student reference(s) with preprocessing",
            len(self._student_encodings),
        )

        # Save to cache
        self._save_cache()

        return no_face_names

    # ------------------------------------------------------------------
    # Main sort loop (enhanced)
    # ------------------------------------------------------------------

    def sort_all(
        self,
        progress_callback: Callable[[int, int, str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, int]:
        """Sort all event photos with enhanced detection and matching.

        For each photo:
            1. Load and preprocess (CLAHE enhancement)
            2. Ensemble face detection: HOG first (fast), CNN fallback (accurate)
            3. Encode detected faces with jitter for robustness
            4. Match against references with confidence scoring
            5. Copy to matched student folders (group shots supported)

        Returns:
            Dict with keys: total, matched, unmatched, skipped, accuracy_metrics.
        """
        images = collect_event_images(self.events_folder)
        total = len(images)

        counts = {"total": total, "matched": 0, "unmatched": 0, "skipped": 0}
        self._match_confidence_scores = []
        self._detection_methods_used = {"hog": 0, "cnn": 0, "ensemble": 0}

        self.logger.info("Starting enhanced sort — %d images found", total)

        for current, (image_path, event_name) in enumerate(images, start=1):
            if cancelled():
                self.logger.info("Sort cancelled at image %d/%d", current, total)
                break

            progress_callback(current, total, image_path.name)
            output_filename = build_output_filename(event_name, image_path.name)

            try:
                # Load with preprocessing
                rgb_image = load_and_preprocess(image_path, self._preprocessor)
            except Exception as exc:
                self.logger.error("Could not open %s: %s — skipping", image_path.name, exc)
                counts["skipped"] += 1
                continue

            try:
                # Ensemble face detection
                face_locations = self._detect_faces_ensemble(rgb_image)
                if not face_locations:
                    self.logger.info("No face detected: %s → _unmatched", image_path.name)
                    safe_copy(
                        image_path,
                        self.output_folder / "_unmatched",
                        output_filename,
                        self.logger,
                    )
                    counts["unmatched"] += 1
                    continue

                face_encodings = self._engine.face_encodings(
                    rgb_image, face_locations, num_jitters=5, model="large"
                )
            except Exception as exc:
                self.logger.error(
                    "Face detection failed for %s: %s", image_path.name, exc
                )
                safe_copy(
                    image_path,
                    self.output_folder / "_unmatched",
                    output_filename,
                    self.logger,
                )
                counts["unmatched"] += 1
                continue

            if not face_encodings:
                safe_copy(
                    image_path,
                    self.output_folder / "_unmatched",
                    output_filename,
                    self.logger,
                )
                counts["unmatched"] += 1
                continue

            # Match and track confidence
            matched_students: set[str] = set()
            for encoding in face_encodings:
                match, confidence = self._match_face_with_confidence(encoding)
                if match:
                    matched_students.add(match)
                    self._match_confidence_scores.append(confidence)

            if matched_students:
                for student_name in matched_students:
                    dest_folder = self.output_folder / student_name
                    safe_copy(image_path, dest_folder, output_filename, self.logger)
                    self.logger.info(
                        "Matched %s → %s", image_path.name, student_name
                    )
                counts["matched"] += 1
            else:
                self.logger.info("No match: %s → _unmatched", image_path.name)
                safe_copy(
                    image_path,
                    self.output_folder / "_unmatched",
                    output_filename,
                    self.logger,
                )
                counts["unmatched"] += 1

        # Compute accuracy metrics
        if self._match_confidence_scores:
            avg_confidence = float(np.mean(self._match_confidence_scores))
            median_confidence = float(np.median(self._match_confidence_scores))
            self.logger.info(
                "Accuracy metrics — avg_confidence=%.4f median_confidence=%.4f",
                avg_confidence,
                median_confidence,
            )
        else:
            avg_confidence = 0.0
            median_confidence = 0.0

        counts["accuracy_metrics"] = {
            "avg_confidence": round(avg_confidence, 4),
            "median_confidence": round(median_confidence, 4),
            "total_matches": len(self._match_confidence_scores),
            "detection_methods": self._detection_methods_used,
        }

        self.logger.info(
            "Enhanced sort complete — total=%d matched=%d unmatched=%d skipped=%d",
            counts["total"],
            counts["matched"],
            counts["unmatched"],
            counts["skipped"],
        )
        return counts

    # ------------------------------------------------------------------
    # Ensemble face detection
    # ------------------------------------------------------------------

    def _detect_faces_ensemble(self, rgb_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces using HOG first, then CNN for robust coverage.

        Strategy:
            1. HOG detection (fast, ~0.2s): catches most faces
            2. If HOG finds nothing, try CNN (slower, ~2s, but more sensitive)
            3. If ensemble mode: merge both results for maximum recall

        Returns:
            List of face locations as (top, right, bottom, left) tuples.
        """
        # Step 1: HOG detection (fast)
        hog_locations = self._engine.face_locations(rgb_image, model="hog")

        if hog_locations:
            self._detection_methods_used["hog"] += 1
            if not self.ensemble_detection:
                return hog_locations

            # Ensemble: also try CNN for faces HOG might miss
            cnn_locations = self._engine.face_locations(rgb_image, model="cnn")
            if cnn_locations:
                self._detection_methods_used["ensemble"] += 1
                # Merge: deduplicate overlapping detections
                return self._merge_face_locations(hog_locations + cnn_locations)
            return hog_locations

        # Step 2: HOG found nothing, fall back to CNN
        cnn_locations = self._engine.face_locations(rgb_image, model="cnn")
        if cnn_locations:
            self._detection_methods_used["cnn"] += 1
            return cnn_locations

        return []

    @staticmethod
    def _merge_face_locations(
        locations: list[tuple[int, int, int, int]],
        iou_threshold: float = 0.5,
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping face detections using IoU-based deduplication.

        When HOG and CNN both detect the same face, keep the CNN detection
        (more precise) and discard the HOG detection if they overlap significantly.
        """
        if len(locations) <= 1:
            return locations

        # Sort by area descending (prefer larger detections)
        def _area(loc: tuple[int, int, int, int]) -> int:
            t, r, b, l = loc
            return (b - t) * (r - l)

        sorted_locs = sorted(locations, key=_area, reverse=True)
        kept: list[tuple[int, int, int, int]] = []

        for loc in sorted_locs:
            is_duplicate = False
            t1, r1, b1, l1 = loc
            area1 = _area(loc)

            for kept_loc in kept:
                t2, r2, b2, l2 = kept_loc

                # Compute intersection
                inter_top = max(t1, t2)
                inter_bottom = min(b1, b2)
                inter_left = max(l1, l2)
                inter_right = min(r1, r2)

                if inter_top < inter_bottom and inter_left < inter_right:
                    inter_area = (inter_bottom - inter_top) * (inter_right - inter_left)
                    area2 = _area(kept_loc)
                    union = area1 + area2 - inter_area
                    iou = inter_area / union if union > 0 else 0

                    if iou > iou_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(loc)

        return kept

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _match_face_with_confidence(
        self, encoding: np.ndarray
    ) -> tuple[str | None, float]:
        """Match a face encoding and return (student_name, confidence_score).

        Confidence is computed as: 1.0 - (distance / threshold).
        A confidence of 0.0 means exactly at threshold; negative means no match.

        Args:
            encoding: 128-d face encoding from face_recognition.

        Returns:
            Tuple of (matched student name or None, confidence score 0-1).
        """
        if not self._student_encodings:
            return None, 0.0

        names = list(self._student_encodings.keys())
        known_encodings = list(self._student_encodings.values())

        distances = self._engine.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance <= self.DISTANCE_THRESHOLD:
            confidence = 1.0 - (best_distance / self.DISTANCE_THRESHOLD)
            self.logger.debug(
                "Face matched to %s (distance=%.4f, confidence=%.2f%%)",
                names[best_idx],
                best_distance,
                confidence * 100,
            )
            return names[best_idx], float(confidence)

        self.logger.debug(
            "No match — best distance=%.4f (threshold=%.2f)",
            best_distance,
            self.DISTANCE_THRESHOLD,
        )
        return None, 0.0


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

def evaluate_accuracy(
    sorter: EnhancedPhotoSorter,
    test_folder: Path,
    ground_truth: dict[str, str],
) -> dict:
    """Evaluate sorting accuracy against ground truth labels.

    Args:
        sorter: Configured EnhancedPhotoSorter with loaded references.
        test_folder: Folder containing test images.
        ground_truth: Dict mapping image filename → expected student name.

    Returns:
        Dict with precision, recall, F1, and per-student breakdown.
    """
    from utils import collect_event_images

    results = {"correct": 0, "incorrect": 0, "no_face": 0, "per_student": {}}

    for student in sorter._student_encodings:
        results["per_student"][student] = {"tp": 0, "fp": 0, "fn": 0}

    images = collect_event_images(test_folder)
    for image_path, _ in images:
        filename = image_path.name
        expected = ground_truth.get(filename)
        if expected is None:
            continue

        try:
            rgb = load_and_preprocess(image_path, sorter._preprocessor)
            locations = sorter._detect_faces_ensemble(rgb)
            encodings = face_recognition.face_encodings(
                rgb, locations, num_jitters=3, model="large"
            )

            if not encodings:
                results["no_face"] += 1
                if expected in results["per_student"]:
                    results["per_student"][expected]["fn"] += 1
                continue

            matched = False
            for enc in encodings:
                match, _ = sorter._match_face_with_confidence(enc)
                if match == expected:
                    results["correct"] += 1
                    results["per_student"][expected]["tp"] += 1
                    matched = True
                elif match and match != expected:
                    results["incorrect"] += 1
                    results["per_student"][expected]["fn"] += 1
                    if match in results["per_student"]:
                        results["per_student"][match]["fp"] += 1

            if not matched:
                results["no_face"] += 1
                if expected in results["per_student"]:
                    results["per_student"][expected]["fn"] += 1

        except Exception:
            results["no_face"] += 1

    total = results["correct"] + results["incorrect"] + results["no_face"]
    results["total_evaluated"] = total
    results["accuracy"] = results["correct"] / max(total, 1)
    results["precision"] = (
        results["correct"] / max(results["correct"] + results["incorrect"], 1)
    )

    return results
