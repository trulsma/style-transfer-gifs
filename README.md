
## Style transfer gifs
Create gifs of images transitioning between different styles

### Example

Output

<img src="turtle.gif" alt="" width="400" height="400" />

Code
```python
content_path = 'turtle.jpg'
style_paths = ['styles/skrik.jpg', 'styles/starry_night.jpg']

create_gif(content_path, style_paths, num_iterations_per_style=300, output="turtle", frametime=100, frames_per_style=25)
```

Content image

<img src="turtle.jpg" alt="" width="400" height="400" />

Image of Green Sea Turtle by P.Lindgren, from [Wikipedia Commons](https://commons.wikimedia.org/wiki/File:Green_Sea_Turtle_grazing_seagrass.jpg)

Style image 1

<img src="styles/skrik.jpg" alt="" width="400" height="400" />

Style image 2

<img src="styles/starry_night.jpg" alt="" width="400" height="400" />


### Use other pictures
Switch 
```content_path = '{your image here}'``` to use a different content image

Switch ```style_paths = ['{style image}', .. more styles if you want]``` to use your style images

Switch ```create_gif(content_path, style_paths, num_iterations_per_style={iterations}, output="{output file}", frametime={time per frame in ms}, frames_per_style={number of styles per frame})```

[Based on this article](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
