from django.db import models


class CodeList(models.Model):
    url = models.URLField(unique = True)
    items = models.TextField()

    def set_items(self, items):
        self.items = ",".join(map(str, items))

    def get_items(self):
        return list(map(int, self.items.split(',')))

    def __str__(self):
        return self.url



class Embedding(models.Model):
    string = models.CharField(max_length=255)
    embedding = models.TextField()
    collection_name = models.CharField(max_length=255)


    def __str__(self):
        return self.string