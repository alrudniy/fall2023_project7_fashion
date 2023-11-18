from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

# Here you should put the path of your image
# IMAGE_PATH = "./DSSI Fashion/image.jpg"

# This is the order of the categories list. NO NOT CHANGE. Just for visualization purposes
cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']
def fix_channels(t):
    """
    Some images may have 4 channels (transparent images) or just 1 channel (black and white images), in order to let the images have only 3 channels. I am going to remove the fourth channel in transparent images and stack the single channel in back and white images.
    :param t: Tensor-like image
    :return: Tensor-like image with three channels
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

def idx_to_text(i):
    return cats[i]
# Random colors used for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        ax.text(xmin, ymin, idx_to_text(cl), fontsize=10,
                bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    
    # Convert the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    # Clear the plot for the next use
    plt.clf()
    
    return img_buffer
    
    
def visualize_predictions(image, outputs, threshold=0.8):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    # plot results
    img_buffer = plot_results(image, probas[keep], bboxes_scaled)
    return img_buffer

def process_image(input_image):
    # Perform image processing here
    MODEL_NAME = "valentinafeve/yolos-fashionpedia"
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

    #image = Image.open(open(IMAGE_PATH, "rb"))
    image = fix_channels(ToTensor()(input_image))
    image = image.resize((600, 800))
    # image

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    img = visualize_predictions(image, outputs, threshold=0.5)
    return img

@app.route("/")
def hello_world():
    return """<div style='height: 600px; 
            width: 800px; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            border: 1px solid black; '>
            <h1 style='color:navy'> 
            This is a backend for apparel items image recognition app
            </h1> </div>"""
    
import base64

@app.route('/process_image', methods=['POST'])
def process_image_route():
    try:
        # Check if the request contains a file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Get the file from the request
        image_file = request.files['image']

        # Check if the file has an allowed extension (you can modify this as needed)
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' not in image_file.filename or image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format'}), 400

        # Read the image from the file
        image = Image.open(BytesIO(image_file.read()))

        # Process the image
        processed_image_buffer = process_image(image)

        # Encode the BytesIO object as a base64 string
        processed_image_base64 = base64.b64encode(processed_image_buffer.getvalue()).decode('utf-8')

        return jsonify({'result': True, 'processed_image': processed_image_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    