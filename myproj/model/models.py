from django.db import models

class Patient(models.Model):
    chol_level = models.IntegerField()
    glu_lvl = models.IntegerField()
    hdl_glu = models.IntegerField()
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    height_cms = models.FloatField()
    weight_kgs = models.FloatField()
    sys_bp = models.IntegerField()
    dia_bp = models.IntegerField()
    waist = models.CharField(max_length=10)
    hip = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Patient {self.id}"
