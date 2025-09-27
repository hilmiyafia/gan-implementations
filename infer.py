
import os
import cv2
import torch
import numpy
from model_stylegan import Generator

import tkinter as tk

if __name__ == "__main__":

    LATENT_DIM = 8
    NOISE_DIM = 16

    model = Generator(LATENT_DIM, NOISE_DIM).cuda()
    base_path = "lightning_logs/version_0/checkpoints/"
    checkpoint = base_path + os.listdir(base_path)[0]
    states = torch.load(checkpoint)["state_dict"]
    states = {
        k.replace("model.", ""): v 
        for k, v in states.items() 
        if k.startswith("model.")}
    model.load_state_dict(states)

    root = tk.Tk()
    root.title("Sliders")

    sliders = []

    def on_slider_change(value):
        with torch.no_grad():
            noises = []
            for rs in sliders:
                b = torch.FloatTensor([s.get() for s in rs])[None]
                noises.append(torch.concat((b, torch.zeros(1, NOISE_DIM)), 1).cuda())
            output = model.generate_with_styles(noises).cpu()
            output = torch.nn.functional.upsample(output, (512, 512))[0]
            output = output.transpose(0, 1).transpose(1, 2)
            output = output * 0.5 + 0.5
            output = torch.flip(output, (2,))
            cv2.imshow("Image", output.numpy())

    def randomize(sliders):
        def run():
            for s in sliders:
                s.set(numpy.random.rand() * 4 - 2)
        return run
    
    def reset(sliders):
        def run():
            for s in sliders:
                s.set(0)
        return run

    for r in range(4):  # 3 rows
        sliders.append([])
        group = tk.LabelFrame(root, text=f"Scale {r + 1}")
        group.grid(row=r, column=0)
        for c in range(LATENT_DIM):  # 8 columns
            slider = tk.Scale(group,
                      from_=-3,
                      to=3,
                      orient='horizontal',
                      showvalue=0,
                      length=100,
                      resolution=0.1,
                      command=on_slider_change)
            slider.grid(row=c // 8, column=c % 8, padx=1, pady=1)
            sliders[-1].append(slider)
        random_button = tk.Button(group, text="Randomize", command=randomize(sliders[-1]))
        random_button.grid(row=0, column=8, padx=1, pady=1)
        reset_button = tk.Button(group, text="Reset", command=reset(sliders[-1]))
        reset_button.grid(row=0, column=9, padx=1, pady=1)

    root.mainloop()
