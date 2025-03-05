from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from PIL import Image
import io
from .utils import extract_text_from_image, extract_nic_details

@api_view(['POST'])
def extract_nic(request):
    try:
        
        front_image_file = request.FILES.get('front_image')
        back_image_file = request.FILES.get('back_image')

        if not front_image_file or not back_image_file:
            return Response({'error': 'Both front and back images are required'}, status=400)

        # Convert to PIL images
        front_image = Image.open(io.BytesIO(front_image_file.read()))
        back_image = Image.open(io.BytesIO(back_image_file.read()))

        # Extract text
        front_text = extract_text_from_image(front_image)
        back_text = extract_text_from_image(back_image)

        return Response({
            'front_side_text': front_text,
            'back_side_text': back_text
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)