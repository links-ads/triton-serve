from triton_serve.config import get_settings
from triton_serve.factory import create_app

app = create_app(get_settings())
