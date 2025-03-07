from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from PIL import Image
import io
from .utils import extract_nic_details  # Assuming extract_nic_details is imported correctly

@api_view(['POST'])
def extract_nic(request):
    try:
        # Get image files from request
        front_image_file = request.FILES.get('front_image')
        back_image_file = request.FILES.get('back_image')

        # Validate image files
        if not front_image_file or not back_image_file:
            return Response({
                'error': 'Both front and back images are required'
            }, status=400)

        # Convert to PIL images
        front_image = Image.open(io.BytesIO(front_image_file.read()))
        back_image = Image.open(io.BytesIO(back_image_file.read()))

        # Extract NIC details from the images
        nic_details = extract_nic_details(front_image, back_image)

        # Return structured response
        return Response({
            'success': True,
            'details': nic_details
        })

    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)
