"""Dos blaze por serial: XYZ e intensidad real, con pypylon 4.2.0.

Basado en reply.py, feria_electrica/live.py y SimpleGrab de Basler.
Las lecturas concurrentes NO sincronizan las exposiciones de las camaras.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import argparse
import time

import numpy as np
from pypylon import pylon


BLAZE_DEVICE_CLASS = "BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras"


import numpy as np


class ToFFrame:
    """"Ovjeto que contiene una PC y una imagen de intensidad, método para convertir a Escala de gris"""
    def __init__(
        self,
        serial: str,
        xyz_mm: np.ndarray,
        gray16: np.ndarray,
        confidence16: np.ndarray,
        host_received_ns: int,
        sensor_timestamp_raw: int,
    ):
        self.serial = serial
        self.xyz_mm = xyz_mm
        self.gray16 = gray16
        self.confidence16 = confidence16
        self.host_received_ns = host_received_ns
        self.sensor_timestamp_raw = sensor_timestamp_raw
    
    def gray8_preview(self) -> np.ndarray:
        """Conversión a escala de grises"""
        image = self.gray16.astype(np.float32)
        low, high = float(image.min()), float(image.max())
        if high <= low:
            return np.zeros(image.shape, dtype=np.uint8)
        return np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


class BlazeToF:
    """Objeto cámara, uno por cada una. No llamar read() concurrentemente sobre el mismo objeto."""

    def __init__(self, serial: str, channel: int, timeout_ms: int = 3000):
        if not serial:
            raise ValueError("Especifique el serial de la camara.")
        if channel not in range(7):
            raise ValueError("MultiCameraChannel debe estar entre 0 y 6.")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms debe ser positivo.")
        self.serial = str(serial)
        self.channel = channel
        self.timeout_ms = timeout_ms
        self._camera = None
        self._old_channel = None

    def open(self) -> "BlazeToF":
        if self._camera is not None:
            raise RuntimeError("La camara ya esta abierta.")

        # DeviceInfo es un FILTRO: CreateFirstDevice resuelve clase + serial.
        info = pylon.DeviceInfo()
        info.SetDeviceClass(BLAZE_DEVICE_CLASS)
        info.SetSerialNumber(self.serial)
        factory = pylon.TlFactory.GetInstance()
        self._camera = pylon.InstantCamera(factory.CreateFirstDevice(info))
        try:
            print(f"Cámara: {info}")
            camera = self._camera
            camera.Open()
            if camera.GetDeviceInfo().GetSerialNumber() != self.serial:
                raise RuntimeError("El serial abierto no coincide con el solicitado.")

            # PixelFormat se configura DESPUES de seleccionar cada componente.
            for name, pixel_format in (
                ("Range", "Coord3D_ABC32f"),
                ("Intensity", "Mono16"),
                ("Confidence", "Confidence16"),
            ):
                camera.ComponentSelector.Value = name
                camera.ComponentEnable.Value = True
                camera.PixelFormat.Value = pixel_format

            # Asi lo configura SimpleGrab de Basler para pypylon + DataContainer.
            camera.GenDCStreamingMode.Value = "On"
            self._old_channel = camera.MultiCameraChannel.Value
            camera.MultiCameraChannel.Value = self.channel
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except BaseException:
            self.close()
            raise
        return self

    def read(self) -> ToFFrame:
        camera = self._camera
        if camera is None or not camera.IsGrabbing():
            raise RuntimeError("Primero abra la camara con open() o with.")

        grab = camera.RetrieveResult(
            self.timeout_ms, pylon.TimeoutHandling_ThrowException,
        )
        try:
            if not grab.GrabSucceeded():
                raise RuntimeError(f"{self.serial}: {grab.ErrorDescription}")
            host_received_ns = time.monotonic_ns()
            sensor_timestamp_raw = int(grab.GetTimeStamp())
            container = grab.GetDataContainer()
            arrays = {}
            for index in range(container.DataComponentCount):
                component = container.GetDataComponent(index)
                try:
                    kind = component.ComponentType
                    height, width = int(component.Height), int(component.Width)
                    if kind == pylon.ComponentType_Range:
                        key, shape = "xyz_mm", (height, width, 3)
                    elif kind == pylon.ComponentType_Intensity:
                        key, shape = "gray16", (height, width)
                    elif kind == pylon.ComponentType_Confidence:
                        key, shape = "confidence16", (height, width)
                    else:
                        continue
                    # Es memoria del SDK: hay que copiar ANTES de ambos Release().
                    arrays[key] = np.array(component.Array, copy=True).reshape(shape)
                finally:
                    component.Release()

            missing = {"xyz_mm", "gray16", "confidence16"} - arrays.keys()
            if missing:
                raise RuntimeError(f"{self.serial}: faltan componentes {sorted(missing)}")
            shape = arrays["xyz_mm"].shape[:2]
            if arrays["gray16"].shape != shape or arrays["confidence16"].shape != shape:
                raise RuntimeError("Los componentes no tienen la misma resolucion.")
            return ToFFrame(
                serial=self.serial, **arrays,
                host_received_ns=host_received_ns,
                sensor_timestamp_raw=sensor_timestamp_raw,
            )
        finally:
            grab.Release()

    def close(self) -> None:
        camera, self._camera = self._camera, None
        old_channel, self._old_channel = self._old_channel, None
        if camera is None:
            return
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        finally:
            try:
                if camera.IsOpen() and old_channel is not None:
                    camera.MultiCameraChannel.Value = old_channel
            finally:
                camera.Close()

    def __enter__(self) -> "BlazeToF":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def list_blaze_devices() -> list[tuple[str, str]]:
    """Enumera dispositivos; no abre ni empieza adquisicion."""
    return [
        (item.GetSerialNumber(), item.GetModelName())
        for item in pylon.TlFactory.GetInstance().EnumerateDevices()
        if item.GetDeviceClass() == BLAZE_DEVICE_CLASS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Listar seriales blaze")
    parser.add_argument("--serials", nargs=2, metavar=("SERIAL_A", "SERIAL_B"))
    parser.add_argument("--channels", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--frames", type=int, default=100, help="Pares de lecturas")
    parser.add_argument("--preview", action="store_true", help="Mostrar ambas intensidades")
    args = parser.parse_args()
    if args.list:
        devices = list_blaze_devices()
        for serial, model in devices:
            print(f"{serial}: {model}")
        if not devices:
            print("No se detectaron camaras blaze.")
        return
    if not args.serials or len(set(args.serials)) != 2:
        parser.error("Indique --serials con dos seriales distintos.")
    if len(set(args.channels)) != 2 or any(ch not in range(7) for ch in args.channels):
        parser.error("Indique dos canales distintos entre 0 y 6.")
    if args.frames <= 0:
        parser.error("--frames debe ser positivo.")

    cv2 = None
    if args.preview:
        import cv2

    try:
        # ExitStack cierra tambien la primera si falla la apertura de la segunda.
        with ExitStack() as stack:
            cameras = [
                stack.enter_context(BlazeToF(serial, channel))
                for serial, channel in zip(args.serials, args.channels)
            ]
            # El executor termina ANTES de cerrar las camaras (orden inverso del stack).
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
            print("Dos flujos activos. Lecturas concurrentes, sin sincronizacion PTP.")
            for sequence in range(args.frames):
                # Enviar AMBAS tareas antes de esperar sus resultados.
                pending = [pool.submit(camera.read) for camera in cameras]
                frame_a, frame_b = [future.result() for future in pending]
                for frame in (frame_a, frame_b):
                    if sequence == 0:
                        print(
                            f"{frame.serial}: xyz={frame.xyz_mm.shape}/{frame.xyz_mm.dtype}, "
                            f"gray={frame.gray16.shape}/{frame.gray16.dtype}, "
                            f"confidence={frame.confidence16.shape}/{frame.confidence16.dtype}"
                        )
                    if cv2 is not None:
                        cv2.imshow(f"Intensity {frame.serial}", frame.gray8_preview())
                if cv2 is not None and cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
