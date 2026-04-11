from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rust_bridge import aggregate_ratings_csv, has_rust_extension, matmul_f32
from rust_bridge import build_user_preferences_from_csv


class RustWrapperTests(unittest.TestCase):
    def test_matmul_matches_numpy(self) -> None:
        rng = np.random.default_rng(123)
        a = rng.normal(size=(5, 7)).astype(np.float32)
        b = rng.normal(size=(7, 3)).astype(np.float32)

        got = matmul_f32(a, b)
        exp = a @ b

        np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-5)

    def test_aggregate_ratings_csv_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "ratings.csv"
            csv_path.write_text(
                "userId,movieId,rating,timestamp\n"
                "1,1,4.0,10\n"
                "2,1,2.0,11\n"
                "3,2,3.5,12\n"
                "4,9,5.0,13\n",
                encoding="utf-8",
            )

            cnt, sm, max_uid = aggregate_ratings_csv(csv_path, max_movie_id=3)

            self.assertEqual(max_uid, 4)
            np.testing.assert_array_equal(cnt, np.array([0, 2, 1, 0], dtype=np.int64))
            np.testing.assert_allclose(sm, np.array([0.0, 6.0, 3.5, 0.0], dtype=np.float64))

    def test_extension_present_for_acceleration(self) -> None:
        self.assertTrue(
            has_rust_extension(),
            "Rust extension not installed. Build with: pip install -e ./rust_ext",
        )

    def test_build_user_preferences_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ratings_csv = td_path / "ratings.csv"
            genome_csv = td_path / "genome-scores.csv"

            ratings_csv.write_text(
                "userId,movieId,rating,timestamp\n"
                "1,1,5.0,10\n"
                "1,2,4.0,11\n"
                "2,1,3.0,12\n",
                encoding="utf-8",
            )
            genome_csv.write_text(
                "movieId,tagId,relevance\n"
                "1,1,1.0\n"
                "1,2,0.5\n"
                "2,1,0.2\n"
                "2,2,1.0\n",
                encoding="utf-8",
            )

            user_ids, prefs = build_user_preferences_from_csv(
                ratings_csv_path=ratings_csv,
                genome_scores_csv_path=genome_csv,
                max_active_users=10,
                rating_scale=0.2,
                num_movies=2,
                num_tags=2,
            )

            np.testing.assert_array_equal(user_ids, np.array([1, 2], dtype=np.int32))
            self.assertEqual(prefs.shape, (2, 3))

            # User 1:
            # tag1 = (1.0 + 0.16) / (1.0 + 0.2) = 0.9666667
            # tag2 = (0.5 + 0.8) / (0.5 + 1.0) = 0.8666667
            np.testing.assert_allclose(prefs[0, 1], 0.9666667, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(prefs[0, 2], 0.8666667, rtol=1e-5, atol=1e-5)

            # User 2 only rated movie1 with scaled rating 0.6
            np.testing.assert_allclose(prefs[1, 1], 0.6, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(prefs[1, 2], 0.6, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
