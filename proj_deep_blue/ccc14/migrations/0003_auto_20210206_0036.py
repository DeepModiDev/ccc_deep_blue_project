# Generated by Django 3.1.4 on 2021-02-05 19:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ccc14', '0002_auto_20210205_2315'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='images',
            name='title',
        ),
        migrations.RemoveField(
            model_name='videos',
            name='title',
        ),
    ]
