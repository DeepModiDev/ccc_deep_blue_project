# Generated by Django 3.1.4 on 2021-02-22 10:23

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ccc14', '0010_auto_20210221_1936'),
    ]

    operations = [
        migrations.RenameField(
            model_name='images',
            old_name='timestamp',
            new_name='date',
        ),
    ]
