from Styles import StyleFrom
from diffusers import DiffusionPipeline


class PipelineWrapper:
    def __init__(self, style_type : int):
        self.style_type = StyleFrom(style_type)
        
    def get_pipeline_with_style(self):
        pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        if self.style_type != 'DEFAULT':
            pipeline.load_lora_weights(self.style_type)
            
        return pipeline