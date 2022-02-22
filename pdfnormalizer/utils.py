def array_to_data(array):
    from PIL import Image
    from io import BytesIO
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data


