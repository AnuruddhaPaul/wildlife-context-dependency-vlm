import ipykernel, nbclient, nbformat, nbconvert, jupyter_client, IPython, torch

print("ipykernel", ipykernel.__version__, "| nbclient", nbclient.__version__, "| nbformat", nbformat.__version__,
      "| nbconvert", nbconvert.__version__, "| IPython", IPython.__version__, "| torch", torch.__version__)
from jupyter_client.kernelspec import KernelSpecManager

print("kernel specs:", list(KernelSpecManager().find_kernel_specs()) or "none registered (native python3 will be used)")
print(KernelSpecManager().get_kernel_spec("python3").argv[:3])
