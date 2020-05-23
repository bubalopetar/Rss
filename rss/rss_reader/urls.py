from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('clickbaits/', views.clickbaits, name='index'),
    path('compare/',views.compare,name='compare')
]
