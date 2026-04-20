from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
import model
import numpy as np


app = Flask(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


DiDCE_net = model.enhance_net_nopool().to(device)
DiDCE_net.load_state_dict(torch.load('./Epoch100.pth', map_location=device))
DiDCE_net.eval()



@app.route("/",methods=["GET","POST"])
def index_route():
    return "Server is listening."

@app.route("/enhance", methods=["POST"])
def enhance_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        low_light_image = Image.open(file.stream).convert("RGB")

        # Image pre processing
        low_light_image = (np.asarray(low_light_image)/255.0)
        low_light_image = torch.from_numpy(low_light_image).float()
        low_light_image = low_light_image.permute(2,0,1)
        low_light_image = low_light_image.to(device).unsqueeze(0)


        # enhanced output Image
        with torch.no_grad():
            enhanced_image, _ = DiDCE_net(low_light_image)

        # Tensor -> image
        output = enhanced_image.squeeze(0)
        output = output.permute(1, 2, 0)
        output = output.clamp(0, 1)
        output = output.detach().cpu().numpy()
        output = (output * 255).astype(np.uint8)

        output_img = Image.fromarray(output)

        # Save to memory
        buffer = io.BytesIO()
        output_img.save(buffer, format="JPEG")
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="enhanced.jpg"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)