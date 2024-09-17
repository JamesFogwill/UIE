Underwater Image Enhancement MSc dissertation project
To run the training for yourself, go to the main file with all the dependent files downloaded too and run.
This will run the SimpleUnet that doesnt not have dual Encoders for RGB and HSV colour space.
If you want to run the RGBHSV Diffusion model then you must comment out the current model and uncomment the RGBHSV model near the top of the main file. This is only available to train. there is no state dict for this model yet.

If you want to test your trained model or the preset one goto the Test.py and replace the checkpoint path with the path to your saved model state dict
Then run the Test.py file. It will generate images and metrics and upload them to tensor board as well as save them in the local directory.
This directory for me is "../", "results", "test_images", "generated.jpg"

you can ignore the WIP file that has alot of my old code through the ages of my dissertation.

Some code in here has been adapted from code written by https://github.com/dome272/Diffusion-Models-pytorch
It was very helpful for a base starting point before i created my model, Thank you!
