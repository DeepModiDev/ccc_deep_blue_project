# Generated by Django 3.1.4 on 2021-03-01 12:59

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('ccc14', '0012_auto_20210223_1641'),
    ]

    operations = [
        migrations.AddField(
            model_name='images',
            name='imageTitle',
            field=models.CharField(default=django.utils.timezone.now, max_length=200),
            preserve_default=False,
        ),
    ]
