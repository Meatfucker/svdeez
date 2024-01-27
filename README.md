# svdeez

A simple gui to interact with Stable Video Diffusion. Image goes on left, gif comes out on right. If you like the output hit accumulate to gather those frames.
 
# Requirements

python 3.10

torch - If you are on windows go use the pytorch install command configurator on the website or else you wont have gpu acceleration

tkinter

customtkinter

opencv-python (maybe its still in my env but might not be needed anymore, still learning tkinter shit)

diffusers

transformers

pillow

probably some other shit I forgot, if it whines about a missing library when you go to run it, just install it.

# Known issue
 Generated gif display will corrupt after saving or accumulating frames, internal frames are still consistent, only display is jacked up.
