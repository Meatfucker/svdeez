import customtkinter
from tkinter import filedialog
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image, ImageTk

class SvdeezGUI(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("SvdeezGUI")
        self.geometry("800x1000")
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        self.svd_pipe.enable_model_cpu_offload()
        self.seed = 42
        self.fps = 12
        self.motion_bucket_id = 180
        self.noise_aug_strength = 0.1
        self.decode_chunk_size = 8
        self.image = None
        self.display_photo = None
        self.frames = []
        self.savepath = "generate.gif"

        self.image_label = customtkinter.CTkLabel(self, text="IMAGE")  # Label to display the image
        self.image_label.grid(row=0, column=0, padx=5, pady=5, sticky="ew", columnspan=4)
        self.load_image_button = customtkinter.CTkButton(self, text="Select Image", command=self.load_init_image)
        self.load_image_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew", columnspan=4)

        self.seed_label = customtkinter.CTkLabel(self, text="Seed:")
        self.seed_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.seed_entry = customtkinter.CTkEntry(self)
        self.seed_entry.grid(row=2, column=0, padx=5, pady=5, sticky="e")

        self.fps_label = customtkinter.CTkLabel(self, text="Fps:")
        self.fps_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.fps_entry = customtkinter.CTkEntry(self)
        self.fps_entry.grid(row=2, column=1, padx=5, pady=5, sticky="e")

        self.motion_bucket_id_label = customtkinter.CTkLabel(self, text="Bucket:")
        self.motion_bucket_id_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.motion_bucket_id_entry = customtkinter.CTkEntry(self)
        self.motion_bucket_id_entry.grid(row=2, column=2, padx=5, pady=5, sticky="e")

        self.decode_chunk_label = customtkinter.CTkLabel(self, text="Chunk:")
        self.decode_chunk_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.decode_chunk_entry = customtkinter.CTkEntry(self)
        self.decode_chunk_entry.grid(row=3, column=0, padx=5, pady=5, sticky="e")

        self.noise_aug_label = customtkinter.CTkLabel(self, text="Noise:")
        self.noise_aug_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.noise_aug_entry = customtkinter.CTkEntry(self)
        self.noise_aug_entry.grid(row=3, column=1, padx=5, pady=5, sticky="e")

        self.save_name_label = customtkinter.CTkLabel(self, text="Name:")
        self.save_name_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.save_name_entry = customtkinter.CTkEntry(self)
        self.save_name_entry.grid(row=3, column=2, padx=5, pady=5, sticky="e")

        self.generate_button = customtkinter.CTkButton(self, text="Generate", command=self.generate)
        self.generate_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew", columnspan=4)


    def generate(self):

        user_seed = self.seed_entry.get()
        user_fps = self.fps_entry.get()
        user_motion_bucket_id = self.motion_bucket_id_entry.get()
        user_decode_chunk = self.decode_chunk_entry.get()
        user_noise = self.noise_aug_entry.get()
        user_save_name = self.save_name_entry.get()

        if user_seed:
            self.seed = int(user_seed)
        if user_fps:
            self.fps = int(user_fps)
        if user_motion_bucket_id:
            self.motion_bucket_id = int(user_motion_bucket_id)
        if user_decode_chunk:
            self.decode_chunk_size = int(user_decode_chunk)
        if user_noise:
            self.noise_aug_strength = float(user_noise)
        if user_save_name:
            self.savepath = f"{user_save_name}.gif"

        generator = torch.manual_seed(self.seed)
        self.frames = self.svd_pipe(self.image, decode_chunk_size=self.decode_chunk_size, generator=generator, motion_bucket_id=self.motion_bucket_id, noise_aug_strength=self.noise_aug_strength).frames[0]
        self.display_frames_as_gif()
        self.frames[0].save(self.savepath, save_all=True, append_images=self.frames[1:], optimize=False, duration=int(1000 / self.fps), loop=0)

    def load_init_image(self):
        imagepath = filedialog.askopenfilename()
        self.image = load_image(imagepath)
        self.image = self.image.resize((1024, 576))
        self.display_photo = customtkinter.CTkImage(light_image=self.image, dark_image=self.image, size=(800, 461))
        self.image_label.configure(image=self.display_photo)

    def display_frames_as_gif(self):
        if self.frames and self.fps:
            gif_label = customtkinter.CTkLabel(self)
            gif_label.grid(row=5, column=0, padx=5, pady=5, sticky="ew", columnspan=4)

            def update_label(index):
                if index < len(self.frames):
                    frame = self.frames[index]
                    photo = ImageTk.PhotoImage(frame)
                    gif_label.configure(image=photo)
                    gif_label.image = photo  # Keep a reference to avoid garbage collection
                    self.after(int(1000 / self.fps), update_label, index + 1)
                else:
                    update_label(0)

            update_label(0)

svdeez = SvdeezGUI()
svdeez.mainloop()