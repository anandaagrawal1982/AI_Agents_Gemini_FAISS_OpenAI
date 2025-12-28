### Image URL Setup
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


#### Environment Setup
import os
os.environ['OPENAI_API_KEY'] = str("xxxxx")

#### Model Initialization
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

### The httpx library fetches the image from the URL, and the base64 
### library encodes the image data in base64 format. 
### This conversion is necessary to send the image data as part of the 
### input to the mode. 

import base64
import httpx
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")


### Constructing the message. This creates a HumanMessage object that contains
### two types of input: a text prompt asking to describe the weather in the 
### image, and the image itself in base64 format.
message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)


### The model.invoke method sends the message to the model,
###  which processes it and returns a response. 
response = model.invoke([message])

### The result, which should be a description of the weather in the image, 
### is then printed out.
print(response.content)