"""Capture the real local RoadEye UI for project presentation slides.

Start the server on http://127.0.0.1:8001 first. Uses a headless browser and
does not change server code or dataset images.
"""

from pathlib import Path

from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright


OUT = Path(__file__).resolve().parent / "presentation_assets"
OUT.mkdir(exist_ok=True)


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto("http://127.0.0.1:8001", wait_until="networkidle", timeout=45_000)
        page.locator("#modelsTable tr").first.wait_for(timeout=30_000)

        page.locator("#runExample").click()
        try:
            page.locator("#exampleResult:not(.hidden)").wait_for(timeout=150_000)
            print("Captured a completed example inference.")
        except PlaywrightTimeout:
            print("Example inference did not finish in time; capturing the interface without a new result.")
        page.locator("#upload-workspace").screenshot(path=str(OUT / "webui_image_inference.png"))

        page.locator('[data-mode="video"]').click()
        page.locator("#upload-workspace").screenshot(path=str(OUT / "webui_video_upload.png"))

        page.locator('[data-mode="live"]').click()
        page.locator("#liveKind").select_option("usb")
        page.locator("#upload-workspace").screenshot(path=str(OUT / "webui_live_camera.png"))

        page.locator(".model-section").screenshot(path=str(OUT / "webui_model_register_full.png"))
        browser.close()

    full = OUT / "webui_model_register_full.png"
    with Image.open(full) as image:
        image.crop((0, 0, image.width, min(560, image.height))).save(OUT / "webui_model_register.png")
    for name in ("webui_image_inference.png", "webui_video_upload.png", "webui_live_camera.png", "webui_model_register.png"):
        path = OUT / name
        print(f"{path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
