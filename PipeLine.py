import os

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn
import timm

class OcrPredict:
    def __init__(self, data_transformer, yolo_model, crnn_model, device, idx_to_char):
        self.data_transformer = data_transformer
        self.yolo_model = yolo_model
        self.crnn_model = crnn_model
        self.device = device
        self.idx_to_char = idx_to_char

    def decode(self, encoded_sequences, blank_char="-"):
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded = []
            for idx in seq:
                if idx != 0:
                    char = self.idx_to_char[idx.item()]
                    if char != blank_char:
                        decoded.append(char)
            decoded_sequences.append("".join(decoded))
        return decoded_sequences

    # Text Detection using Yolov8
    def text_detection(self, img_path):
        result = self.yolo_model(img_path, verbose=False)[0]
        bboxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        names = result.names
        confs = result.boxes.conf.tolist()
        return bboxes, classes, names, confs

    # Text Recognition using CRNN
    def text_recognition(self, img, transform):
        transformed_img = transform(img)
        transformed_img = transformed_img.unsqueeze(0).to(self.device)

        self.crnn_model.eval()
        with torch.no_grad():
            logits = self.crnn_model(transformed_img).detach().cpu()

        text = self.decode(logits.permute(1, 0, 2).argmax(dim=2))
        return text

    # Visualization function
    def visualize(self, img, detections):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')

        for bbox, detected_class, confidence, text in detections:
            x1, y1, x2, y2 = bbox
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False,
                    edgecolor='red',
                    linewidth=2,
                )
            )
            plt.text(x1, y1 - 10, f"{detected_class} ({confidence:.2f}): {text}", fontsize=10,
                     bbox=dict(facecolor='red', alpha=0.5))

        plt.show()

    # Prediction function
    def predict(self, img_path):
        bboxes, classes, names, confs = self.text_detection(img_path)
        img = Image.open(img_path).convert('RGB')
        pred = []

        for bbox, cls, conf in zip(bboxes, classes, confs):
            x1, y1, x2, y2 = bbox
            confidence = conf
            name = names[int(cls)]
            cropped = img.crop((x1, y1, x2, y2))

            trans_text = self.text_recognition(
                cropped,
                self.data_transformer
            )
            pred.append((bbox, name, confidence, trans_text))

        self.visualize(img, pred)

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layer=3):
        super(CRNN, self).__init__()

        backbone = timm.create_model('resnet101', in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        for param in self.backbone[-unfreeze_layer:].parameters():
            param.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            512,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2))

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)
        return x



def main(img_folder):

    chars = "012345678abcdefghijklmnopqrstuvwxyz-"
    vocab_size = len(chars)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ])

    crnn_path = "best_crnn_model.pth"
    yolo_path = "runs\\detect\\train42\\weights\\best.pt"
    yolo_model = YOLO(yolo_path)
    crnn_model = CRNN(vocab_size=vocab_size, hidden_size=512, n_layers=2, dropout=0.2).to(device)
    crnn_model.load_state_dict(torch.load(crnn_path))

    ocr_predict = OcrPredict(data_transforms, yolo_model, crnn_model, device, idx_to_char)

    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        ocr_predict.predict(img_path)

if __name__ == '__main__':
    main("Dataset/ryoungt_05.08.2002")