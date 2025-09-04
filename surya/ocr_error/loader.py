# from typing import Optional

# import torch

# from surya.common.load import ModelLoader
# from surya.logging import get_logger
# from surya.ocr_error.model.config import DistilBertConfig
# from surya.ocr_error.model.encoder import DistilBertForSequenceClassification
# from surya.ocr_error.tokenizer import DistilBertTokenizer
# from surya.settings import settings

# logger = get_logger()


# class OCRErrorModelLoader(ModelLoader):
#     def __init__(self, checkpoint: Optional[str] = None):
#         super().__init__(checkpoint)

#         if self.checkpoint is None:
#             self.checkpoint = settings.OCR_ERROR_MODEL_CHECKPOINT

#     def model(
#         self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
#     ) -> DistilBertForSequenceClassification:
#         if device is None:
#             device = settings.TORCH_DEVICE_MODEL
#         if dtype is None:
#             dtype = settings.MODEL_DTYPE

#         config = DistilBertConfig.from_pretrained(self.checkpoint)
#         model = (
#             DistilBertForSequenceClassification.from_pretrained(
#                 self.checkpoint,
#                 torch_dtype=dtype,
#                 config=config,
#             )
#             .to(device)
#             .eval()
#         )

#         if settings.COMPILE_ALL or settings.COMPILE_OCR_ERROR:
#             torch.set_float32_matmul_precision("high")
#             torch._dynamo.config.cache_size_limit = 1
#             torch._dynamo.config.suppress_errors = False

#             logger.info(
#                 f"Compiling detection model {self.checkpoint} from {DistilBertForSequenceClassification.get_local_path(self.checkpoint)} onto device {device} with dtype {dtype}"
#             )
#             compile_args = {"backend": "openxla"} if device == "xla" else {}
#             model = torch.compile(model, **compile_args)

#         return model

#     def processor(
#         self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
#     ) -> DistilBertTokenizer:
#         return DistilBertTokenizer.from_pretrained(self.checkpoint)
# ocr_error/loader.py (updated with IPEX frontend integration)
from typing import Optional

import torch

# NEW: guarded IPEX import
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except Exception:
    HAS_IPEX = False

from surya.common.load import ModelLoader
from surya.logging import get_logger
from surya.ocr_error.model.config import DistilBertConfig
from surya.ocr_error.model.encoder import DistilBertForSequenceClassification
from surya.ocr_error.tokenizer import DistilBertTokenizer
from surya.settings import settings

logger = get_logger()

class OCRErrorModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.OCR_ERROR_MODEL_CHECKPOINT

    def model(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> DistilBertForSequenceClassification:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = DistilBertConfig.from_pretrained(self.checkpoint)
        model = (
            DistilBertForSequenceClassification.from_pretrained(
                self.checkpoint,
                torch_dtype=dtype,
                config=config,
            )
            .to(device)
            .eval()
        )

        # NEW: IPEX frontend optimization (eager mode only)
        model._ipex_enabled = False
        model._ipex_bf16 = False

        # Apply only on CPU
        on_cpu = str(device).startswith("cpu")
        if HAS_IPEX and on_cpu:
            try:
                # BF16 preferred with weights_prepack=False
                model = ipex.optimize(
                    model,
                    dtype=torch.bfloat16,
                    inplace=True,
                    weights_prepack=False
                )
                model._ipex_enabled = True
                model._ipex_bf16 = True
            except Exception:
                # FP32 fallback
                try:
                    model = ipex.optimize(
                        model,
                        dtype=torch.float,
                        inplace=True,
                        weights_prepack=False
                    )
                    model._ipex_enabled = True
                    model._ipex_bf16 = False
                except Exception:
                    # Skip if all fail; continue as original
                    pass

        if settings.COMPILE_ALL or settings.COMPILE_OCR_ERROR:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            logger.info(
                f"Compiling detection model {self.checkpoint} from {DistilBertForSequenceClassification.get_local_path(self.checkpoint)} onto device {device} with dtype {dtype}"
            )
            compile_args = {"backend": "openxla"} if device == "xla" else {}
            model = torch.compile(model, **compile_args)

        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> DistilBertTokenizer:
        return DistilBertTokenizer.from_pretrained(self.checkpoint)
