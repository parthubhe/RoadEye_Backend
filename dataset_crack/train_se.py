"""Convenience launcher for the squeeze-and-excitation YOLO26 ablation."""

try:
    from train_attention import main
except ModuleNotFoundError:  # pragma: no cover - module invocation
    from dataset_crack.train_attention import main


if __name__ == "__main__":
    main("se")
