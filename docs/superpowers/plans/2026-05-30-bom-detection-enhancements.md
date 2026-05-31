# BOM Pattern Detection System - Final Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deep cancellation handling, a dynamically-loaded presets library, a HuggingFace deployment entrypoint, and a comprehensive project specification.

**Architecture:** We use a cooperative interruption model where a thread-safe `CancellationState` is checked inside heavy loop structures in `engines.py` and `detector.py`, raising a `DetectionCancelledException` to gracefully stop computation and release VRAM/CPU. We integrate Gradio's native event cancellation for UI responsiveness, build a dynamic directory scanning callback for presets, place a root-level `app.py` entrypoint, and compile a premium technical spec in `SPECIFICATION.md`.

**Tech Stack:** Python, Gradio, PyTorch, OpenCV, pytest

---

## Proposed Changes Mapping

### Component: Core Architecture (Cancellation)
We introduce custom cancellation exceptions and a shared state to allow cross-thread interrupt signals.

#### [MODIFY] [exceptions.py](file:///d:/CV_BOM_Detection/src/exceptions.py)
*   Add `DetectionCancelledException` inheriting from `BOMDetectorException`.
*   Add the thread-safe `CancellationState` sharing class.

#### [MODIFY] [engines.py](file:///d:/CV_BOM_Detection/src/engines.py)
*   Update `multiscale_template_match` to support `cancellation_state` checking within the scale loop.

#### [MODIFY] [detector.py](file:///d:/CV_BOM_Detection/src/detector.py)
*   Update `detect` to support `cancellation_state` checking and passing down to `multiscale_template_match`.

### Component: UI & Integration
We add preset drop-downs, dynamic file discovery, cancel buttons, and graceful error handling.

#### [MODIFY] [app.py](file:///d:/CV_BOM_Detection/src/app.py)
*   Implement dynamic directory listing for `./data/drawings/` and `./data/patterns/`.
*   Add dropdown preset elements and their event hooks.
*   Add the "Cancel Detection" button, Gradio queue state, and `DetectionCancelledException` handler.

### Component: Entrypoint & Docs
We provide entrypoints for HuggingFace Spaces and write technical specifications.

#### [NEW] [app.py (Root)](file:///d:/CV_BOM_Detection/app.py)
*   Create root-level entrypoint that launches the Gradio dashboard with queuing enabled.

#### [NEW] [SPECIFICATION.md (Root)](file:///d:/CV_BOM_Detection/SPECIFICATION.md)
*   Create the comprehensive technical report on problem analysis, hybrid architecture comparison, flows, and benchmark framework.

---

## Detailed Tasks

### Task 1: Custom Exceptions & Cancellation State

**Files:**
- Modify: [exceptions.py](file:///d:/CV_BOM_Detection/src/exceptions.py)
- Modify: [tests/test_detector.py](file:///d:/CV_BOM_Detection/tests/test_detector.py)

- [ ] **Step 1: Write a failing unit test for `CancellationState`**
  Modify [tests/test_detector.py](file:///d:/CV_BOM_Detection/tests/test_detector.py) by adding this test function at the end:
  ```python
  def test_cancellation_state_toggles():
      from src.exceptions import CancellationState
      state = CancellationState()
      assert not state.is_cancelled
      state.cancel()
      assert state.is_cancelled
      state.reset()
      assert not state.is_cancelled
  ```

- [ ] **Step 2: Run the test to verify it fails due to missing imports**
  Run: `python -m pytest tests/test_detector.py -k test_cancellation_state_toggles`
  Expected: Fail with `ImportError: cannot import name 'CancellationState'`

- [ ] **Step 3: Implement custom exception and `CancellationState` class**
  Modify [exceptions.py](file:///d:/CV_BOM_Detection/src/exceptions.py) by appending:
  ```python
  class DetectionCancelledException(BOMDetectorException):
      """Ngoại lệ ném ra khi người dùng huỷ quá trình phát hiện."""
      pass

  class CancellationState:
      """Trạng thái hủy đồng bộ giữa luồng giao diện và luồng tính toán."""
      def __init__(self):
          self.is_cancelled = False
          
      def cancel(self):
          self.is_cancelled = True
          
      def reset(self):
          self.is_cancelled = False
  ```

- [ ] **Step 4: Run the test to verify it passes**
  Run: `python -m pytest tests/test_detector.py -k test_cancellation_state_toggles`
  Expected: PASS

- [ ] **Step 5: Commit changes**
  ```bash
  git add src/exceptions.py tests/test_detector.py
  git commit -m "feat: add DetectionCancelledException and CancellationState"
  ```

---

### Task 2: Engine Integration

**Files:**
- Modify: [engines.py](file:///d:/CV_BOM_Detection/src/engines.py)
- Modify: [tests/test_engines.py](file:///d:/CV_BOM_Detection/tests/test_engines.py)

- [ ] **Step 1: Write a failing unit test for cancelled matching**
  Modify [tests/test_engines.py](file:///d:/CV_BOM_Detection/tests/test_engines.py) by adding this test at the end:
  ```python
  def test_multiscale_template_match_cancellation():
      from src.engines import multiscale_template_match
      from src.exceptions import CancellationState, DetectionCancelledException
      
      drawing = np.ones((100, 100), dtype=np.uint8) * 255
      tmpl = np.ones((10, 10), dtype=np.uint8) * 255
      
      state = CancellationState()
      state.cancel()  # Immediately cancel
      
      with pytest.raises(DetectionCancelledException):
          multiscale_template_match(
              drawing, tmpl, scale_range=(0.8, 1.2), scale_step=0.1, cancellation_state=state
          )
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_engines.py -k test_multiscale_template_match_cancellation`
  Expected: Fail with `Failed: DID NOT RAISE <class 'src.exceptions.DetectionCancelledException'>`

- [ ] **Step 3: Modify `multiscale_template_match` to support cancellation checks**
  Modify the signature of `multiscale_template_match` in [engines.py](file:///d:/CV_BOM_Detection/src/engines.py) to accept `cancellation_state=None`, and check it in the scale loop:
  ```python
  def multiscale_template_match(
      drawing_gray: np.ndarray,
      template_preprocessed: np.ndarray,
      scale_range: Tuple[float, float] = (0.5, 1.5),
      scale_step: float = 0.05,
      threshold: float = 0.50,
      cancellation_state: Any = None,
  ) -> List[Tuple[int, int, int, int, float, float]]:
      ...
      for scale in scales:
          if cancellation_state and cancellation_state.is_cancelled:
              from src.exceptions import DetectionCancelledException
              raise DetectionCancelledException("Quá trình quét ảnh đã bị người dùng hủy.")
          ...
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_engines.py -k test_multiscale_template_match_cancellation`
  Expected: PASS

- [ ] **Step 5: Commit changes**
  ```bash
  git add src/engines.py tests/test_engines.py
  git commit -m "feat: integrate cancellation checks into multiscale_template_match"
  ```

---

### Task 3: Detector Orchestrator Integration

**Files:**
- Modify: [detector.py](file:///d:/CV_BOM_Detection/src/detector.py)
- Modify: [tests/test_detector.py](file:///d:/CV_BOM_Detection/tests/test_detector.py)

- [ ] **Step 1: Write a failing test for cancelled detection**
  Modify [tests/test_detector.py](file:///d:/CV_BOM_Detection/tests/test_detector.py) by adding this test at the end:
  ```python
  def test_detector_cancellation(dummy_grayscale_drawing, dummy_pattern):
      from src.detector import PatternDetector
      from src.exceptions import CancellationState, DetectionCancelledException
      
      detector = PatternDetector(device="cpu")
      detector.load_drawing(dummy_grayscale_drawing)
      detector.add_templates([dummy_pattern])
      
      state = CancellationState()
      state.cancel()
      
      with pytest.raises(DetectionCancelledException):
          detector.detect(cancellation_state=state)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_detector.py -k test_detector_cancellation`
  Expected: Fail with `Failed: DID NOT RAISE <class 'src.exceptions.DetectionCancelledException'>`

- [ ] **Step 3: Modify `PatternDetector.detect` to accept and check `cancellation_state`**
  Modify the `detect` signature in [detector.py](file:///d:/CV_BOM_Detection/src/detector.py) to accept `cancellation_state: Any = None`. Add checks at the beginning of `detect` and within the rotation loops, passing it down to `multiscale_template_match`:
  ```python
      def detect(
          self,
          mode: str = "v3",
          confidence_threshold: float = 0.75,
          v1_threshold: float = 0.50,
          v2_threshold: float = 0.80,
          alpha: float = 0.30,
          iou_threshold: float = 0.30,
          enable_local_refine: bool = False,
          variance_std_threshold: float = 5.0,
          context_margin_pct: float = 0.15,
          extractor_type: str = "auto",
          cancellation_state: Any = None,
      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
          ...
          if self.drawing_gray is None:
              raise BOMDetectorException("Drawing chưa được nạp.")
          if not self.templates_variants:
              raise BOMDetectorException("Template chưa được đăng ký.")
              
          if cancellation_state and cancellation_state.is_cancelled:
              from src.exceptions import DetectionCancelledException
              raise DetectionCancelledException("Quá trình quét ảnh đã bị người dùng hủy.")
          ...
          for idx, (tmpl, rotation_name) in enumerate(self.templates_variants):
              if cancellation_state and cancellation_state.is_cancelled:
                  from src.exceptions import DetectionCancelledException
                  raise DetectionCancelledException("Quá trình quét ảnh đã bị người dùng hủy.")
              ...
              if mode == "v1":
                  ...
                  proposals = multiscale_template_match(
                      drawing_edge, tmpl_edge, threshold=v1_threshold, cancellation_state=cancellation_state
                  )
              elif mode == "v2":
                  proposals = multiscale_template_match(
                      drawing_edge, tmpl_edge, threshold=0.35, cancellation_state=cancellation_state
                  )
              elif mode == "v3":
                  proposals = multiscale_template_match(
                      drawing_edge, tmpl_edge, threshold=v1_threshold, cancellation_state=cancellation_state
                  )
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_detector.py -k test_detector_cancellation`
  Expected: PASS

- [ ] **Step 5: Run all unit tests to ensure no regressions**
  Run: `python -m pytest -v`
  Expected: All 38+ tests PASS

- [ ] **Step 6: Commit changes**
  ```bash
  git add src/detector.py tests/test_detector.py
  git commit -m "feat: propagate cancellation_state from PatternDetector.detect"
  ```

---

### Task 4: Dynamic Preset Library Integration

**Files:**
- Modify: [app.py](file:///d:/CV_BOM_Detection/src/app.py)
- Modify: [tests/test_app.py](file:///d:/CV_BOM_Detection/tests/test_app.py)

- [ ] **Step 1: Write a unit test verifying dynamic loading of presets**
  Modify [tests/test_app.py](file:///d:/CV_BOM_Detection/tests/test_app.py) by appending:
  ```python
  def test_preset_file_discovery():
      import os
      from src.app import discover_presets
      
      patterns, drawings = discover_presets()
      # Verify lists contain only valid extensions or are lists
      assert isinstance(patterns, list)
      assert isinstance(drawings, list)
      
      for p in patterns:
          assert p.lower().endswith(('.png', '.jpg', '.jpeg'))
      for d in drawings:
          assert d.lower().endswith(('.png', '.jpg', '.jpeg'))
  ```

- [ ] **Step 2: Run the test to verify it fails due to missing function**
  Run: `python -m pytest tests/test_app.py -k test_preset_file_discovery`
  Expected: Fail with `ImportError: cannot import name 'discover_presets'`

- [ ] **Step 3: Implement preset discovery and selector elements**
  Modify [app.py](file:///d:/CV_BOM_Detection/src/app.py):
  1. Add `discover_presets` function to find files in `./data/patterns/` and `./data/drawings/` (relative to workspace root):
     ```python
     def discover_presets() -> tuple[list[str], list[str]]:
         import os
         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         patterns_dir = os.path.join(base_dir, "data", "patterns")
         drawings_dir = os.path.join(base_dir, "data", "drawings")
         
         valid_exts = ('.png', '.jpg', '.jpeg')
         
         patterns = []
         if os.path.exists(patterns_dir):
             patterns = sorted([f for f in os.listdir(patterns_dir) if f.lower().endswith(valid_exts)])
             
         drawings = []
         if os.path.exists(drawings_dir):
             drawings = sorted([f for f in os.listdir(drawings_dir) if f.lower().endswith(valid_exts)])
             
         return patterns, drawings
     ```
  2. Implement `load_preset_image` callback:
     ```python
     def load_preset_image(filename: str, subfolder: str) -> Union[str, None]:
         if not filename:
             return None
         import os
         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         return os.path.join(base_dir, "data", subfolder, filename)
     ```
  3. Integrate into the Gradio Layout in `with gr.Blocks(title="Zero-Shot BOM Pattern Detector Pro") as demo:`:
     - Under `pattern_input` and `drawing_input`, add:
       ```python
       preset_patterns, preset_drawings = discover_presets()
       
       with gr.Accordion("💡 Preset Sample Library (Thư viện mẫu sẵn)", open=False):
           pattern_preset = gr.Dropdown(
               choices=[""] + preset_patterns, 
               value="", 
               label="Select Pattern Preset (Chọn mẫu sẵn)"
           )
           drawing_preset = gr.Dropdown(
               choices=[""] + preset_drawings, 
               value="", 
               label="Select Drawing Preset (Chọn bản vẽ sẵn)"
           )
       ```
     - Wire dropdown changes:
       ```python
       pattern_preset.change(
           fn=lambda name: load_preset_image(name, "patterns"),
           inputs=[pattern_preset],
           outputs=[pattern_input]
       )
       drawing_preset.change(
           fn=lambda name: load_preset_image(name, "drawings"),
           inputs=[drawing_preset],
           outputs=[drawing_input]
       )
       ```

- [ ] **Step 4: Run the test to verify it passes**
  Run: `python -m pytest tests/test_app.py -k test_preset_file_discovery`
  Expected: PASS

- [ ] **Step 5: Commit changes**
  ```bash
  git add src/app.py tests/test_app.py
  git commit -m "feat: add dynamic preset discovery and drop-downs to UI"
  ```

---

### Task 5: UI Cancellation Button & State Integration

**Files:**
- Modify: [app.py](file:///d:/CV_BOM_Detection/src/app.py)
- Modify: [tests/test_app.py](file:///d:/CV_BOM_Detection/tests/test_app.py)

- [ ] **Step 1: Write an integration test for cancellation in Gradio inference**
  Modify [tests/test_app.py](file:///d:/CV_BOM_Detection/tests/test_app.py) by appending:
  ```python
  def test_run_app_inference_cancellation(tmp_path, dummy_pattern, dummy_grayscale_drawing):
      from src.exceptions import CancellationState
      pattern_path = os.path.join(tmp_path, "pattern.png")
      drawing_path = os.path.join(tmp_path, "drawing.png")
      cv2.imwrite(pattern_path, dummy_pattern)
      cv2.imwrite(drawing_path, dummy_grayscale_drawing)
      
      cancellation_state = CancellationState()
      cancellation_state.cancel()  # Abort before start
      
      vis, json_out, dashboard_html = run_app_inference(
          pattern_path=pattern_path,
          drawing_path=drawing_path,
          mode="v3",
          conf_thresh=0.40,
          v1_thresh=0.40,
          v2_thresh=0.50,
          alpha=0.30,
          iou_thresh=0.30,
          enable_refine=True,
          var_std=5.0,
          margin=0.15,
          extractor_choice="auto",
          cancellation_state=cancellation_state
      )
      
      assert vis is None
      assert isinstance(json_out, dict)
      assert "error" in json_out
      assert "Quá trình quét ảnh đã bị người dùng hủy" in json_out["error"]
  ```

- [ ] **Step 2: Run the test to verify it fails**
  Run: `python -m pytest tests/test_app.py -k test_run_app_inference_cancellation`
  Expected: Fail because `run_app_inference` does not accept `cancellation_state` or check it.

- [ ] **Step 3: Modify `run_app_inference` and add the Cancellation Button**
  Modify [app.py](file:///d:/CV_BOM_Detection/src/app.py):
  1. Add `cancellation_state: Any = None` parameter to `run_app_inference` signature.
  2. Within `run_app_inference`, reset state:
     ```python
     if cancellation_state is not None:
         cancellation_state.reset()
     ```
  3. Inside the `try` block, pass `cancellation_state` to `detector.detect`:
     ```python
     results, report = detector.detect(
         ...,
         cancellation_state=cancellation_state
     )
     ```
  4. In the `except` blocks, catch `DetectionCancelledException`:
     ```python
     from src.exceptions import DetectionCancelledException
     ...
     except DetectionCancelledException as e:
         return None, {"error": f"Bị hủy: {str(e)}"}, "<div style='color: #e71d36; font-weight: bold;'>❌ Quá trình đã bị hủy bởi người dùng.</div>"
     ```
  5. In the Gradio blocks layout:
     - Define `cancellation_state = gr.State(CancellationState())` inside `with gr.Blocks(...) as demo:`:
       ```python
       cancellation_state = gr.State(value=None)
       ```
       Wait, let's instantiate the state cleanly:
       ```python
       from src.exceptions import CancellationState
       state_helper = gr.State(value=CancellationState())
       ```
     - In the buttons row, add the Cancel button next to the Run button:
       ```python
       with gr.Row():
           run_btn = gr.Button("⚡ Run Detection", variant="primary")
           cancel_btn = gr.Button("❌ Cancel Detection", variant="stop")
       ```
     - Implement `cancel_inference` callback:
       ```python
       def cancel_inference(state: CancellationState) -> str:
           if state is not None:
               state.cancel()
           return "Đang hủy tiến trình..."
       ```
     - Bind click events:
       ```python
       run_event = run_btn.click(
           fn=run_app_inference,
           inputs=[
               pattern_input,
               drawing_input,
               mode_input,
               conf_input,
               v1_input,
               v2_input,
               alpha_input,
               iou_input,
               refine_input,
               var_input,
               margin_input,
               extractor_input,
               state_helper
           ],
           outputs=[
               output_image,
               json_output,
               dashboard_output
           ]
       )
       
       cancel_btn.click(
           fn=cancel_inference,
           inputs=[state_helper],
           outputs=[],
           cancels=[run_event]
       )
       ```

- [ ] **Step 4: Run the test to verify it passes**
  Run: `python -m pytest tests/test_app.py -k test_run_app_inference_cancellation`
  Expected: PASS

- [ ] **Step 5: Run all tests to verify full codebase health**
  Run: `python -m pytest -v`
  Expected: All 40+ tests PASS!

- [ ] **Step 6: Commit changes**
  ```bash
  git add src/app.py tests/test_app.py
  git commit -m "feat: integrate Cancel button and CancellationState into Gradio app"
  ```

---

### Task 6: Root Entrance App Creation

**Files:**
- Create: [app.py](file:///d:/CV_BOM_Detection/app.py)

- [ ] **Step 1: Verify running from root**
  Verify there is no root `app.py`.
  Command: `dir app.py`
  Expected: "File not found"

- [ ] **Step 2: Create root-level `app.py`**
  Write to [app.py](file:///d:/CV_BOM_Detection/app.py):
  ```python
  import sys
  import os

  # Đảm bảo import tuyệt đối từ thư mục src/ hoạt động chính xác
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

  from src.app import demo

  if __name__ == "__main__":
      # Bật hàng chờ (queue) phục vụ cơ chế hủy bất đồng bộ trên Gradio
      demo.queue().launch(
          server_name="0.0.0.0",
          server_port=7860
      )
  ```

- [ ] **Step 3: Verify running the root entrypoint locally**
  Propose to run: `python app.py` (Let it run for 2-3 seconds, then we can interrupt it or verify it starts fine).

- [ ] **Step 4: Commit root app.py**
  ```bash
  git add app.py
  git commit -m "feat: add root-level entrypoint app.py for HuggingFace Space"
  ```

---

### Task 7: Premium Specification & Documentation

**Files:**
- Create: [SPECIFICATION.md](file:///d:/CV_BOM_Detection/SPECIFICATION.md)
- Modify: [README.md](file:///d:/CV_BOM_Detection/README.md)

- [ ] **Step 1: Write `SPECIFICATION.md`**
  Create the premium detailed technical specification matching the requested design exactly. Include all three flowcharts in Mermaid.js.

- [ ] **Step 2: Update `README.md`**
  Modify [README.md](file:///d:/CV_BOM_Detection/README.md) to add:
  - Clear note about the new "❌ Cancel Detection" button and dynamic Presets Sample Library.
  - A dedicated "🤗 Deploying to Hugging Face Spaces" guide detailing:
    1. Setting up a new Gradio Space on Hugging Face.
    2. Checking that `app.py` resides at the root level.
    3. Git commands or manual upload steps to push `src/`, `data/`, `app.py`, and `requirements.txt`.
    4. Explaining why `server_name="0.0.0.0"` is used to bind proxy traffic within Docker.
    5. How to access and share the public HTTPS Space URL.

- [ ] **Step 3: Commit documentation**
  ```bash
  git add SPECIFICATION.md README.md
  git commit -m "docs: write SPECIFICATION.md and update README.md with HF deployment guide"
  ```

---

## Verification Plan

### Automated Tests
- Run `python -m pytest -v` to ensure 100% of unit and integration tests are passing.

### Manual Verification
- Launch `python app.py` locally and verify:
  1. The Preset drop-downs are loaded with drawings and patterns, and selecting one automatically fills the Image uploads.
  2. Clicking "⚡ Run Detection" executes correctly.
  3. Clicking "❌ Cancel Detection" while a detection is running immediately aborts the calculation, shows "Bị hủy" in the output, and restores UI responsiveness.
