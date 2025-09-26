from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),  # <-- root URL
    path('prediction/', views.predict_promotion, name='prediction'),
    path('employee_search/', views.employee_search_view, name='employee_search'),
]

