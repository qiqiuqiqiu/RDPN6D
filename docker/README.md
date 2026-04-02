# Docker Usage

## Build

```bash
docker build -t rdpn6d:local .
```

Or with compose:

```bash
docker compose build
```

## Run

```bash
docker run --gpus all -it --rm \
  --shm-size=16g \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/bus/usb:/dev/bus/usb \
  -v $(pwd):/workspace/RDPN6D \
  -v $(pwd)/datasets:/workspace/RDPN6D/datasets \
  -v $(pwd)/output:/workspace/RDPN6D/output \
  -w /workspace/RDPN6D \
  rdpn6d:local
```

Or:

```bash
docker compose run --rm rdpn6d
```

## RealSense Demo

Inside the container:

```bash
python tools/demo/realsense_realtime_demo.py \
  --config-file configs/gdrn/lm/a6_cPnP_lm13.py \
  --weights output/gdrn/lm/residual_0831_32/model_final.pth \
  --obj-name ape
```

Press `s` to select the target ROI once, then the demo will keep tracking and estimating pose.
Press `r` to reset ROI and `q` to quit.

## Notes

- The image intentionally does not bake `datasets/` and `output/` into the layer history.
- Mount your trained weights into `output/` or another host directory.
- If you deploy on a local machine without GPU, this image is not the right target.
- If your final local machine uses a different CUDA driver, you may need to rebuild from a different CUDA base image.
- If `detectron2` build fails, the most common fix is aligning the CUDA base image with your host NVIDIA driver and PyTorch CUDA wheel.
- For local GUI display on Linux, run `xhost +local:root` on the host before starting the container.
