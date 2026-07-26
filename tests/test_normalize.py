"""Tests for frame/still normalisation and perceptual hashing."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from mkv_episode_matcher.normalize import (
    NORMALIZED_SIZE,
    crop_borders,
    hamming_distance,
    hamming_distances,
    hash_array,
    normalize_array,
    normalize_image,
    pack_hashes,
)

from .conftest import make_pattern


class TestCropBorders:
    def test_removes_letterbox_bars(self):
        content = make_pattern(3, width=200, height=100)
        framed = np.zeros((200, 200), dtype=np.uint8)
        framed[50:150, :] = content

        cropped = crop_borders(framed)

        assert cropped.shape == (100, 200)
        np.testing.assert_array_equal(cropped, content)

    def test_removes_pillarbox_bars(self):
        content = make_pattern(4, width=100, height=200)
        framed = np.zeros((200, 200), dtype=np.uint8)
        framed[:, 50:150] = content

        cropped = crop_borders(framed)

        assert cropped.shape == (200, 100)

    def test_leaves_borderless_image_untouched(self):
        image = make_pattern(5, width=64, height=64)

        np.testing.assert_array_equal(crop_borders(image), image)

    def test_all_black_image_is_returned_unchanged(self):
        image = np.zeros((32, 32), dtype=np.uint8)

        assert crop_borders(image).shape == (32, 32)

    def test_never_crops_more_than_the_allowed_fraction(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        image[48:52, 48:52] = 255

        cropped = crop_borders(image, max_crop_fraction=0.3)

        assert cropped.shape[0] >= 40
        assert cropped.shape[1] >= 40

    def test_tolerates_near_black_bars(self):
        content = make_pattern(6, width=120, height=60)
        framed = np.full((120, 120), 6, dtype=np.uint8)
        framed[30:90, :] = content

        assert crop_borders(framed, black_level=16).shape == (60, 120)


class TestNormalizeArray:
    def test_output_is_canonical_size_and_grayscale(self):
        result = normalize_array(make_pattern(1, width=1920, height=1080))

        assert result.size == NORMALIZED_SIZE
        assert result.mode == "L"

    def test_normalizes_rgb_images(self):
        rgb = Image.fromarray(make_pattern(2)).convert("RGB")

        assert normalize_image(rgb).mode == "L"


class TestHashStability:
    def test_rescaled_image_keeps_a_near_identical_hash(self):
        original = Image.fromarray(make_pattern(11, width=1280, height=720))
        rescaled = original.resize((640, 360), Image.LANCZOS)

        a = hash_array(np.asarray(original))
        b = hash_array(np.asarray(rescaled))

        assert hamming_distance(a.phash, b.phash) <= 2

    def test_letterboxed_image_matches_the_unpadded_original(self):
        content = make_pattern(12, width=320, height=180)
        letterboxed = np.zeros((240, 320), dtype=np.uint8)
        letterboxed[30:210, :] = content

        a = hash_array(content)
        b = hash_array(letterboxed)

        assert hamming_distance(a.phash, b.phash) <= 2

    def test_distinct_images_are_far_apart(self):
        a = hash_array(make_pattern(21))
        b = hash_array(make_pattern(22))

        assert hamming_distance(a.phash, b.phash) >= 12

    def test_jpeg_recompression_keeps_the_hash_close(self, tmp_path):
        original = Image.fromarray(make_pattern(13))
        path = tmp_path / "still.jpg"
        original.save(path, quality=60)

        a = hash_array(np.asarray(original))
        b = hash_array(np.asarray(Image.open(path).convert("L")))

        assert hamming_distance(a.phash, b.phash) <= 4


class TestHammingHelpers:
    def test_distance_of_identical_values_is_zero(self):
        assert hamming_distance(0xDEADBEEF, 0xDEADBEEF) == 0

    def test_distance_counts_differing_bits(self):
        assert hamming_distance(0b1010, 0b0001) == 3

    def test_vectorised_distances_match_the_scalar_helper(self):
        values = [0x0, 0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0]
        packed = pack_hashes(values)

        result = hamming_distances(0x0F0F0F0F0F0F0F0F, packed)

        expected = [hamming_distance(0x0F0F0F0F0F0F0F0F, v) for v in values]
        assert result.tolist() == expected

    def test_packing_preserves_full_64_bit_values(self):
        packed = pack_hashes([0xFFFFFFFFFFFFFFFF, 1])

        assert packed.dtype == np.uint64
        assert packed.tolist() == [0xFFFFFFFFFFFFFFFF, 1]

    def test_distances_against_an_empty_index(self):
        assert hamming_distances(1, pack_hashes([])).size == 0

    @given(
        st.integers(min_value=0, max_value=2**64 - 1),
        st.integers(min_value=0, max_value=2**64 - 1),
        st.integers(min_value=0, max_value=2**64 - 1),
    )
    def test_hamming_is_a_metric(self, a, b, c):
        assert hamming_distance(a, b) == hamming_distance(b, a)
        assert (hamming_distance(a, b) == 0) == (a == b)
        assert hamming_distance(a, c) <= hamming_distance(a, b) + hamming_distance(b, c)

    @given(st.lists(st.integers(min_value=0, max_value=2**64 - 1), max_size=20))
    def test_vectorised_distances_are_bounded_by_the_hash_width(self, values):
        distances = hamming_distances(0xAAAAAAAAAAAAAAAA, pack_hashes(values))

        assert all(0 <= d <= 64 for d in distances.tolist())


def test_hash_array_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        hash_array(np.zeros((0, 0), dtype=np.uint8))
