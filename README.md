## Multimodal Inference API for Qwen 2.5 Omni

### Getting Started

#### Using Docker

`sudo docker-compose up --build`

#### Client UI

```
# first change the const API_BASE in index.html to the address of the running server
cd client
python -m http.server
```

#### Models

The default model is Qwen 2.5 Omni but you can also use Phi-4-multimodal with
`MODEL_TYPE=phi docker-compose up --build`

#### Tailscale

Initially to enable connecting to my GPU instance I was using Nginx, which is why docker is configured with it.

But to make connecting from my local machine and the GPU instance, in this case on Lambda Labs, much easier. I just install Tailscale with `curl -fsSL https://tailscale.com/install.sh | sh` then run the login command and auth via the url it provides. At that point it is as if the VM and my laptop are both local so connectivity is simple.

#### Manual Install

```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install accelerate
```
