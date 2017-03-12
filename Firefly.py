import random

class Firefly:
   def __init__(self, f_id, problemSize):
      self.loads = []
      self.id = f_id
      self.problemSize = problemSize
   
   #PGi - power of i-th generator
   def addLoad(self, PGi):
      if (len(self.loads) < self.problemSize):
         self.loads.append(PGi)
      else:
         print("Warning! Attemption to add too much loads.")

   def addRandomLoad(self, PGiMin, PGiMax):
      self.addLoad(PGiMin + random.random() * (PGiMax - PGiMin))

   def deep_copy(self):
      ff = Firefly(self.id, self.problemSize)
      ff.intensity = self.intensity
      ff.loads = self.loads.copy()
      return ff

   def __str__(self):
      return ("intensity=%f" % self.intensity)
