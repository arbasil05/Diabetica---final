from django.contrib import admin
from .models import Patient

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('id', 'chol_level', 'glu_lvl', 'hdl_glu', 'age', 'gender', 'height_cms', 'weight_kgs', 'sys_bp', 'dia_bp', 'waist', 'hip', 'created_at')
    list_filter = ('gender',)
    search_fields = ('id', 'gender')

    def has_add_permission(self, request):
        return False  # Disable the "Add" button to prevent adding new records from the admin interface

    def has_delete_permission(self, request, obj=None):
        return False  # Disable the "Delete" button to prevent deleting records from the admin interface
