from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name = 'index'),
    path('example/<str:parameter>/', views.YourView.as_view(), name = 'example'),
    path('process-repo/', views.process_repo, name = 'process-repo')
]