from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from .models import UpImage, UpVideo
from .forms import UpImageForm, UpVideoForm
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from .functionset import load_image
import dlib
import cv2
import os
import base64
from skimage.transform import resize
import tempfile
import shutil


# Create your views here.

def index(request):
    context = {}
    return render(request, 'pages/index.html', context)


def upload_img(request):
    form = UpImageForm()
    img = UpImage.objects.all()
    prob = 0.0
    context = {}

    if request.method == 'POST':
        form = UpImageForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save()

            face_detector = dlib.get_frontal_face_detector()
            # 이미지 로드 및 처리
            image = cv2.imdecode(np.fromstring(post.image.read(), np.uint8), cv2.IMREAD_COLOR)
            test_image = image.copy()

            # 그레이스케일로 변환
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

            # 얼굴 영역 감지
            faces = face_detector(gray)

            # 모델 로드
            model_path = 'model.h5'
            model = load_model(model_path)

            # 얼굴이 감지된 경우에만 처리
            if len(faces) > 0:
                # 첫 번째 얼굴 영역 선택
                face = faces[0]

                # 얼굴 영역 추출
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                margin = int((right - left) * 0.3)
                left -= margin
                top -= margin
                right += margin
                bottom += margin
                face_img = image[top:bottom, left:right]

                # 이미지 크기를 모델 입력 크기 (380, 380)으로 조정
                face_img = cv2.resize(face_img, (380, 380))

                # 모델로 예측 수행
                input_data = tf.convert_to_tensor(face_img, dtype=tf.float32)
                input_data = tf.image.resize(input_data, (380, 380))
                input_data = tf.expand_dims(input_data, axis=0)
                predictions = model(input_data)
                # 확률 값 추출
                prob = predictions[0][0]

                # 확률 값을 표시할 텍스트 생성
                text = f'Probability: {prob * 100:.2f}%'

                # 네모 박스 그리기
                cv2.rectangle(test_image, (left, top), (right, bottom), (255, 0, 0), 2)

                # 확률 값을 이미지에 표시
                cv2.putText(test_image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                # 특성 맵 시각화
                layer_name = "efficientnetb4"  # 원하는 층의 이름을 지정합니다.

                # 해당 층의 출력을 얻습니다.
                layer = model.get_layer(layer_name)
                feature_map = layer(input_data)

                # 시각화를 위해 첫 번째 이미지 (배치 차원)의 결과만 사용합니다.
                feature_map = feature_map[0]
                prob = float(prob)
                prob = str(round(prob, 3) * 100) + "%"

                # 입력 이미지와 특성 맵을 오버랩하여 시각화합니다.
                resized_feature_map = resize(feature_map.numpy().sum(axis=-1),
                                             (image.shape[0], image.shape[1]))
                alpha = 0.5  # 오버랩 투명도 조절
                overlaid_image = image.copy()
                overlaid_image[:, :, 0] = (1 - alpha) * overlaid_image[:, :, 0] + alpha * resized_feature_map

            with tf.GradientTape() as tape:
                tape.watch(input_data)
                predictions = model(input_data)
                probability = predictions[0][0]


            gradients = tape.gradient(probability, input_data)
            saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)

            # Saliency map을 원래 이미지 위에 오버랩하여 시각화
            saliency_map_normalized = saliency_map[0] / tf.reduce_max(saliency_map[0])  # 정규화

            # Saliency map을 RGB 형식으로 변환하여 오버랩
            saliency_map_rgb = cv2.applyColorMap(np.uint8(255 * saliency_map_normalized), cv2.COLORMAP_JET)

            # 이미지 오버랩
            overlay_image = cv2.addWeighted(cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR), 0.7, saliency_map_rgb, 0.5, 0)
            #overlaid_image = cv2.applyColorMap(overlaid_image, cv2.COLORMAP_JET)

            # 원본 이미지를 저장하거나 출력하려면 적절한 방법을 사용하세요.
            # 여기서는 HttpResponse를 사용하여 이미지를 출력합니다.
            # _, buffer = cv2.imencode('.jpg', image)
            # image_data = base64.b64encode(buffer).decode('utf-8')
            # prob = float(prob)
            # prob = str(round(prob, 3) * 100) + "%"

            # image를 bytestream으로 변환하여 전송
            ret, buffer = cv2.imencode('.png', test_image)
            image_data = base64.b64encode(buffer).decode('utf-8')

            ret, feature_buffer = cv2.imencode('.png', overlay_image)
            overlay_data = base64.b64encode(feature_buffer).decode('utf-8')

            ret, saliency_buffer = cv2.imencode('.png', saliency_map_rgb)
            saliency_data = base64.b64encode(saliency_buffer).decode('utf-8')

            context['image'] = image_data
            context['overlay'] = overlay_data
            context['saliency'] = saliency_data
            context['probability'] = prob
            context['post'] = post.image
            # context['email'] = post.email

    return render(request, 'pages/upload_img.html', context)


def upload_vid(request):
    form = UpVideoForm(request.POST, request.FILES)
    context = {}

    if request.method == 'POST':
        form = UpVideoForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save()
            vid_path = post.video.path
            cap = cv2.VideoCapture(vid_path)
            model_path = 'model.h5'
            model = load_model(model_path)
            face_detector = dlib.get_frontal_face_detector()

            output_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector(gray)

                if len(faces) > 0:
                    for face in faces:
                        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                        margin = int((right - left) * 0.3)
                        left -= margin
                        top -= margin
                        right += margin
                        bottom += margin
                        face_img = frame[top:bottom, left:right]
                        face_img = cv2.resize(face_img, (380, 380))

                        input_data = tf.convert_to_tensor(face_img, dtype=tf.float32)
                        input_data = tf.image.resize(input_data, (380, 380))
                        input_data = tf.expand_dims(input_data, axis=0)

                        predictions = model(input_data)
                        probability = predictions[0][0]
                        text = f'Probability: {probability * 100:.2f}%'
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                output_frames.append(frame_data)

            # 모든 프레임을 하나의 동영상 스트림으로 만들어 HttpResponse로 반환
            response = HttpResponse(b''.join(output_frames), content_type='video/mp4')
            response['Content-Disposition'] = f'attachment; filename="processed_video.mp4"'
            return response

    return render(request, 'pages/upload_vid.html', context)






