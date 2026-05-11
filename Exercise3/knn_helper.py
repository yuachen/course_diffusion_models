"""Nearest-neighbor helpers for the MNIST memorization sanity check.

The notebook keeps the high-level experiment in Section 5. This module holds
the reusable KNN search, exact uint8 membership check, and plotting utilities.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from tqdm import tqdm


KNNResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]


class MNISTLike(Protocol):
    """Minimal interface used from torchvision.datasets.MNIST."""

    data: Any
    targets: Any

    def __len__(self) -> int:
        ...


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    """Map image tensors from [-1, 1] to [0, 1] for plotting or quantization."""
    return (images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0


@torch.no_grad()
def mnist_uint8_images_and_labels(
    mnist_train_dataset: MNISTLike,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MNIST images as uint8 tensors with shape (N, 1, 28, 28), plus labels."""
    if not hasattr(mnist_train_dataset, "data") or not hasattr(mnist_train_dataset, "targets"):
        raise TypeError("This helper expects torchvision.datasets.MNIST.")

    images_uint8 = mnist_train_dataset.data
    if images_uint8.ndim == 3:
        images_uint8 = images_uint8.unsqueeze(1)
    labels = torch.as_tensor(mnist_train_dataset.targets, dtype=torch.long)
    return images_uint8.cpu(), labels.cpu()


@torch.no_grad()
def find_k_nearest_training_images(
    generated_images: torch.Tensor,
    mnist_train_dataset: MNISTLike,
    k: int = 5,
    batch_size: int | None = None,
    search_device: torch.device | None = None,
) -> KNNResult:
    """Find the k nearest MNIST training images by exact batched L2 search.

    Distances are computed in the normalized [-1, 1] pixel scale. The matrix-product
    identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y lets us search all training
    images without a Python loop over individual examples.

    Returns tensors with a neighbor axis of length k:
    nearest_images:  (num_queries, k, 1, 28, 28)
    nearest_labels:  (num_queries, k)
    nearest_indices: (num_queries, k)
    nearest_mse:     (num_queries, k)
    nearest_rmse:    (num_queries, k)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    train_images_uint8, train_labels = mnist_uint8_images_and_labels(mnist_train_dataset)
    num_train = train_images_uint8.shape[0]
    k = min(int(k), num_train)

    if search_device is None:
        search_device = generated_images.device
    if batch_size is None:
        batch_size = 8192 if search_device.type == "cuda" else 2048

    queries = generated_images.detach().clamp(-1.0, 1.0).to(search_device, dtype=torch.float32)
    queries_flat = queries.flatten(1)
    num_queries, num_pixels = queries_flat.shape
    query_norm2 = queries_flat.square().sum(dim=1).view(1, num_queries)

    best_dist2 = torch.full((k, num_queries), float("inf"), device=search_device)
    best_indices = torch.full((k, num_queries), -1, dtype=torch.long, device=search_device)

    for start in tqdm(range(0, num_train, batch_size), desc="Nearest-neighbor chunks"):
        end = min(start + batch_size, num_train)
        train_uint8 = train_images_uint8[start:end].to(search_device, non_blocking=True)
        train_batch = 2.0 * (train_uint8.to(torch.float32) / 255.0) - 1.0
        train_flat = train_batch.flatten(1)

        train_norm2 = train_flat.square().sum(dim=1).view(-1, 1)
        dist2 = train_norm2 + query_norm2 - 2.0 * (train_flat @ queries_flat.T)
        dist2 = dist2.clamp_min(0.0)

        chunk_k = min(k, end - start)
        chunk_best_dist2, chunk_best_pos = torch.topk(dist2, k=chunk_k, dim=0, largest=False)
        chunk_best_indices = start + chunk_best_pos

        combined_dist2 = torch.cat([best_dist2, chunk_best_dist2], dim=0)
        combined_indices = torch.cat([best_indices, chunk_best_indices], dim=0)
        best_dist2, order = torch.topk(combined_dist2, k=k, dim=0, largest=False)
        best_indices = torch.gather(combined_indices, dim=0, index=order)

    nearest_indices = best_indices.T.contiguous().cpu()
    nearest_dist2 = best_dist2.T.contiguous().cpu()

    nearest_uint8 = train_images_uint8[nearest_indices.reshape(-1)]
    nearest_images = 2.0 * (nearest_uint8.float() / 255.0) - 1.0
    nearest_images = nearest_images.view(num_queries, k, *nearest_images.shape[1:])
    nearest_labels = train_labels[nearest_indices.reshape(-1)].view(num_queries, k)
    nearest_mse = nearest_dist2 / num_pixels
    nearest_rmse = torch.sqrt(nearest_mse)

    generated_uint8 = (
        (255.0 * _to_display_range(generated_images[:num_queries]))
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    train_hashes = {image.numpy().tobytes() for image in train_images_uint8}
    exact_uint8_matches = [image.numpy().tobytes() in train_hashes for image in generated_uint8.cpu()]

    return nearest_images, nearest_labels, nearest_indices, nearest_mse, nearest_rmse, exact_uint8_matches


def find_nearest_training_images(
    generated_images: torch.Tensor,
    mnist_train_dataset: MNISTLike,
    batch_size: int | None = None,
    search_device: torch.device | None = None,
) -> KNNResult:
    """Backward-compatible one-nearest-neighbor wrapper."""
    nearest_images, nearest_labels, nearest_indices, nearest_mse, nearest_rmse, exact_uint8_matches = (
        find_k_nearest_training_images(
            generated_images,
            mnist_train_dataset,
            k=1,
            batch_size=batch_size,
            search_device=search_device,
        )
    )
    return (
        nearest_images[:, 0],
        nearest_labels[:, 0],
        nearest_indices[:, 0],
        nearest_mse[:, 0],
        nearest_rmse[:, 0],
        exact_uint8_matches,
    )


def print_k_nearest_summary(
    nearest_labels: torch.Tensor,
    nearest_indices: torch.Tensor,
    nearest_mse: torch.Tensor,
    nearest_rmse: torch.Tensor,
    exact_uint8_matches: list[bool],
) -> None:
    """Print a compact top-k nearest-neighbor table."""
    num_queries, k = nearest_indices.shape
    print("sample | rank | train index | label | pixel MSE | pixel RMSE | generated exact 8-bit match")
    print("-------+------+-------------+-------+-----------+------------+-------------------------------")
    for sample_idx in range(num_queries):
        for rank in range(k):
            exact_text = str(exact_uint8_matches[sample_idx]) if rank == 0 else ""
            print(
                f"{sample_idx:6d} | {rank + 1:4d} | {int(nearest_indices[sample_idx, rank]):11d} | "
                f"{int(nearest_labels[sample_idx, rank]):5d} | "
                f"{float(nearest_mse[sample_idx, rank]):9.6f} | "
                f"{float(nearest_rmse[sample_idx, rank]):10.6f} | {exact_text}"
            )


def show_k_nearest_neighbor_rows(
    generated_images: torch.Tensor,
    nearest_images: torch.Tensor,
    max_rows_per_figure: int = 4,
    title_prefix: str = "Five nearest MNIST training images",
) -> None:
    """Show generated image, its k nearest neighbors, and absolute differences.

    Each row has 1 + 2k panels:
    generated | NN 1 | |generated-NN 1| | ... | NN k | |generated-NN k|.
    Difference panels are black for identical pixels and brighter for larger differences.
    """
    generated = generated_images.detach().clamp(-1.0, 1.0).cpu()
    nearest = nearest_images.detach().clamp(-1.0, 1.0).cpu()
    from matplotlib import pyplot as plt

    if nearest.ndim != 5:
        raise ValueError(f"Expected nearest_images with shape (N, k, C, H, W), got {tuple(nearest.shape)}.")
    if generated.shape[0] != nearest.shape[0]:
        raise ValueError("generated_images and nearest_images must have the same first dimension.")

    num_queries, k = nearest.shape[:2]
    ncols = 1 + 2 * k
    column_titles = ["generated"]
    for rank in range(k):
        column_titles.extend([f"NN {rank + 1}", "|diff|"])

    for start in range(0, num_queries, max_rows_per_figure):
        end = min(start + max_rows_per_figure, num_queries)
        nrows = end - start
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.25 * ncols, 1.25 * nrows), squeeze=False)

        for local_row, sample_idx in enumerate(range(start, end)):
            axes[local_row, 0].imshow(
                _to_display_range(generated[sample_idx]).squeeze(0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[local_row, 0].set_ylabel(f"sample {sample_idx}", fontsize=8)

            for rank in range(k):
                neighbor_col = 1 + 2 * rank
                diff_col = neighbor_col + 1

                axes[local_row, neighbor_col].imshow(
                    _to_display_range(nearest[sample_idx, rank]).squeeze(0),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )

                difference = (generated[sample_idx] - nearest[sample_idx, rank]).abs().squeeze(0) / 2.0
                axes[local_row, diff_col].imshow(difference, cmap="gray", vmin=0.0, vmax=1.0)

            for col in range(ncols):
                axes[local_row, col].set_xticks([])
                axes[local_row, col].set_yticks([])
                for spine in axes[local_row, col].spines.values():
                    spine.set_visible(False)

        for col, title in enumerate(column_titles):
            axes[0, col].set_title(title, fontsize=8)

        fig.suptitle(f"{title_prefix}: generated samples {start} to {end - 1}", y=1.02)
        plt.tight_layout()
        plt.show()


def run_nearest_neighbor_check(
    generated_images: torch.Tensor,
    mnist_train_dataset: MNISTLike,
    k: int = 5,
    search_device: torch.device | None = None,
    max_rows_per_figure: int = 4,
    title_prefix: str = "Nearest-neighbor check: generated image with closest training images",
) -> KNNResult:
    """Run the full nearest-neighbor sanity check and display the results."""
    result = find_k_nearest_training_images(
        generated_images,
        mnist_train_dataset,
        k=k,
        search_device=search_device,
    )
    nearest_images, nearest_labels, nearest_indices, nearest_mse, nearest_rmse, exact_uint8_matches = result

    num_queries = generated_images.shape[0]
    print(f"Searched {len(mnist_train_dataset):,} MNIST training images for each of {num_queries} generated images.")
    print(f"Showing the top {k} nearest neighbors for each generated image.")
    print(f"Exact 8-bit pixel matches after quantization: {sum(exact_uint8_matches)} / {num_queries}")
    print()

    print_k_nearest_summary(
        nearest_labels,
        nearest_indices,
        nearest_mse,
        nearest_rmse,
        exact_uint8_matches,
    )
    show_k_nearest_neighbor_rows(
        generated_images,
        nearest_images,
        max_rows_per_figure=max_rows_per_figure,
        title_prefix=title_prefix,
    )
    return result


__all__ = [
    "find_k_nearest_training_images",
    "find_nearest_training_images",
    "mnist_uint8_images_and_labels",
    "print_k_nearest_summary",
    "run_nearest_neighbor_check",
    "show_k_nearest_neighbor_rows",
]
