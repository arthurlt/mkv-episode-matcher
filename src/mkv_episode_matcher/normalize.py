"""Canonical image normalisation and perceptual hashing.

Every image that enters the matcher -- decoded video frames as well as provider
stills -- passes through :func:`normalize_image` before being hashed. Applying
one shared pipeline to both sides is what makes the Hamming distances in
:mod:`mkv_episode_matcher.score_visual` comparable: a 1920x1080 letterboxed
frame and a 780px wide JPEG still of the same moment must collapse onto the
same canonical form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

__all__ = [
    "NORMALIZED_SIZE",
    "ImageHashes",
    "crop_borders",
    "hamming_distance",
    "hamming_distances",
    "hash_array",
    "hash_image",
    "load_and_hash",
    "normalize_array",
    "normalize_image",
    "pack_hashes",
]

#: Canonical size every image is squashed to before hashing.
NORMALIZED_SIZE = (128, 128)

#: Width of a single perceptual hash, in bits.
HASH_BITS = 64

_HASH_SIZE = 8
_DEFAULT_BLACK_LEVEL = 16
_DEFAULT_MAX_CROP_FRACTION = 0.6


@dataclass(frozen=True, slots=True)
class ImageHashes:
    """A pair of perceptual hashes for one normalised image.

    Attributes
    ----------
    phash
        64-bit DCT-based perceptual hash; the primary matching signal.
    dhash
        64-bit difference hash; a cheap corroborating signal that fails
        differently from ``phash`` on colour-graded promotional stills.
    """

    phash: int
    dhash: int


def crop_borders(
    array: np.ndarray,
    *,
    black_level: int = _DEFAULT_BLACK_LEVEL,
    max_crop_fraction: float = _DEFAULT_MAX_CROP_FRACTION,
) -> np.ndarray:
    """Trim uniform dark letterbox/pillarbox bars from the edges of an image.

    Bars are detected per row and per column: an edge line is considered a bar
    when every pixel in it sits at or below ``black_level``. At most
    ``max_crop_fraction`` of each axis may be removed, so a genuinely dark shot
    is never eroded down to a sliver.

    Parameters
    ----------
    array
        Two-dimensional ``uint8`` grayscale image.
    black_level
        Highest pixel value still considered "black". Slightly above zero to
        survive lossy compression of the bars themselves.
    max_crop_fraction
        Upper bound on the fraction of each axis that may be trimmed.

    Returns
    -------
    numpy.ndarray
        A view of ``array`` with the detected bars removed.

    Examples
    --------
    >>> import numpy as np
    >>> framed = np.zeros((10, 6), dtype=np.uint8)
    >>> framed[3:7, :] = 200
    >>> crop_borders(framed).shape
    (4, 6)
    >>> crop_borders(np.zeros((4, 4), dtype=np.uint8)).shape
    (4, 4)
    """
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {array.shape}")
    if array.size == 0:
        return array

    rows = _content_span(array.max(axis=1), black_level, array.shape[0], max_crop_fraction)
    cols = _content_span(array.max(axis=0), black_level, array.shape[1], max_crop_fraction)
    return array[rows[0] : rows[1], cols[0] : cols[1]]


def _content_span(
    line_maxima: np.ndarray, black_level: int, length: int, max_crop_fraction: float
) -> tuple[int, int]:
    """Return the ``[start, stop)`` span of non-black lines, honouring the crop cap."""
    lit = np.flatnonzero(line_maxima > black_level)
    if lit.size == 0:
        return 0, length

    start, stop = int(lit[0]), int(lit[-1]) + 1
    minimum_keep = max(1, int(round(length * (1.0 - max_crop_fraction))))
    while stop - start < minimum_keep:
        if start > 0:
            start -= 1
        if stop - start < minimum_keep and stop < length:
            stop += 1
    return start, stop


def normalize_array(array: np.ndarray) -> Image.Image:
    """Crop bars from ``array`` and squash it to the canonical hashing form.

    Parameters
    ----------
    array
        Grayscale (2-D) or RGB (3-D) ``uint8`` image.

    Returns
    -------
    PIL.Image.Image
        Grayscale image of size :data:`NORMALIZED_SIZE`.

    Examples
    --------
    >>> import numpy as np
    >>> normalize_array(np.full((90, 160), 128, dtype=np.uint8)).size
    (128, 128)
    """
    if array.ndim == 3:
        array = np.asarray(Image.fromarray(array).convert("L"))
    cropped = crop_borders(array)
    return Image.fromarray(cropped).resize(NORMALIZED_SIZE, Image.LANCZOS)


def normalize_image(image: Image.Image) -> Image.Image:
    """Normalise a :class:`PIL.Image.Image` exactly as decoded frames are.

    Parameters
    ----------
    image
        Source image in any mode.

    Returns
    -------
    PIL.Image.Image
        Grayscale image of size :data:`NORMALIZED_SIZE`.
    """
    return normalize_array(np.asarray(image.convert("L")))


def hash_array(array: np.ndarray) -> ImageHashes:
    """Normalise and perceptually hash a raw image array.

    Parameters
    ----------
    array
        Grayscale or RGB ``uint8`` image.

    Returns
    -------
    ImageHashes
        The perceptual and difference hashes of the normalised image.

    Examples
    --------
    >>> import numpy as np
    >>> flat = hash_array(np.full((64, 64), 200, dtype=np.uint8))
    >>> flat.phash == hash_array(np.full((32, 32), 200, dtype=np.uint8)).phash
    True
    """
    if array.size == 0:
        raise ValueError("cannot hash an empty image array")
    return hash_image_normalized(normalize_array(array))


def hash_image(image: Image.Image) -> ImageHashes:
    """Normalise and perceptually hash a :class:`PIL.Image.Image`."""
    return hash_image_normalized(normalize_image(image))


def hash_image_normalized(image: Image.Image) -> ImageHashes:
    """Hash an already-normalised image without touching it again."""
    return ImageHashes(
        phash=_to_int(imagehash.phash(image, hash_size=_HASH_SIZE)),
        dhash=_to_int(imagehash.dhash(image, hash_size=_HASH_SIZE)),
    )


def load_and_hash(path: Path) -> ImageHashes:
    """Load an image file from disk and return its normalised hashes.

    Parameters
    ----------
    path
        Path to any image format Pillow can decode.

    Returns
    -------
    ImageHashes
        Hashes of the normalised image.
    """
    with Image.open(path) as image:
        return hash_image(image)


def _to_int(value: imagehash.ImageHash) -> int:
    """Convert an :class:`imagehash.ImageHash` bit matrix into a 64-bit integer."""
    return int.from_bytes(np.packbits(value.hash.flatten()).tobytes(), "big")


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two hashes.

    Examples
    --------
    >>> hamming_distance(0b1010, 0b0001)
    3
    >>> hamming_distance(255, 255)
    0
    """
    return (left ^ right).bit_count()


def pack_hashes(values: list[int]) -> np.ndarray:
    """Pack integer hashes into a ``uint64`` array for vectorised comparison.

    Examples
    --------
    >>> pack_hashes([1, 2]).tolist()
    [1, 2]
    >>> pack_hashes([]).size
    0
    """
    return np.asarray(values, dtype=np.uint64).reshape(-1)


def hamming_distances(query: int, packed: np.ndarray) -> np.ndarray:
    """Return the Hamming distance from ``query`` to every hash in ``packed``.

    Parameters
    ----------
    query
        A 64-bit hash.
    packed
        ``uint64`` array of hashes, as produced by :func:`pack_hashes`.

    Returns
    -------
    numpy.ndarray
        ``int16`` array of distances, one per entry in ``packed``.

    Examples
    --------
    >>> hamming_distances(0b1010, pack_hashes([0b1010, 0b0001])).tolist()
    [0, 3]
    """
    if packed.size == 0:
        return np.empty(0, dtype=np.int16)
    return np.bitwise_count(packed ^ np.uint64(query)).astype(np.int16)
