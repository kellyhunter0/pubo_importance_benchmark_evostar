#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

	Walsh expansion
	with basic tools to make computation with Walsh polynomials, and output in json file format

	@authors: LEC group, LISIC Lab, Univ. of Littoral Opal Coast, France
	Created on Fri Apr 30 14:57:22 2021

Please cite this article if you use this code:
  Sara Tari, Sebastien Verel, and Mahmoud Omidvar. 
  "PUBOi: A Tunable Benchmark with Variable Importance." 
  In European Conference on Evolutionary Computation in Combinatorial Optimization (Part of EvoStar), pp. 175-190. Springer, Cham, 2022.

"""
import json
import time

class WalshExpansion:
	def __init__(self, n = 0):
		self.n = n
		# expansion: dictionary of ( tuple: double )
		self.expansion = { }

	def __str__(self):
		return self.export()

	def to_json(self, fileName, generator):
		date_infos = {"date":time.strftime("%X %x %Z"),
			  "generator": "https://gitlab.com/verel/pubo-importance-benchmark/src/generator/puboi_generator.py",
			  "author": "LEC group/LISIC/ULCO"
			  }
		if generator.shift:
			shift = "true"
		else:
			shift = "false"
		res = '{"problem": {"type": "puboi", ' + json.dumps(date_infos)[1:-1] + \
		', "seed":' + str(generator.seed) + ', "n":' + str(generator.n) + \
		', "size":' +str(generator.importance['size']) + ', "degree":' +str(generator.importance['degree']) + \
		', "m":'+str(generator.m)+', "factor":' + str(generator.factor) + \
		', "p_function":' + str(generator.p_function) + ', "typeWeight":' + str(generator.typeWeight) + \
		', "shift":' + shift + \
		', "objective": "' + generator.objective + '", "bound": ' + str(generator.bound) + \
		', "terms": [' + self.export() + ']}}\n'
		json_file = open(fileName, "w")
		json_file.write(res)
		json_file.close()

	def to_json_minimal(self, fileName):
		date_infos = {"date": time.strftime("%X %x %Z") }
		res = '{"problem": {"type": "unkown", ' + json.dumps(date_infos)[1:-1] + \
		', "n":' + str(self.n) + \
		', "terms": [' + self.export() + ']}}\n'
		json_file = open(fileName, "w")
		json_file.write(res)
		json_file.close()

	def load(self, fileName):
		with open(fileName, 'r+') as f:
			data = json.load(f)
			self.n = data['problem']['n']
			for elem in data['problem']['terms']:
				self.addTerm(elem['w'], tuple(elem['ids']))
		
	def addTerm(self, v, ids):
		ve = self.expansion.get(ids)
		if ve == None:
			self.expansion[ids] = v
		else:
			self.expansion[ids] += v

	def mult(self, alpha):
		for k in self.expansion:
			self.expansion[k] *= alpha

	def sum(self, p):
		for k in p.expansion:
			v = self.expansion.get(k) 
			if v == None:
				self.expansion[k] = p.expansion[k]
			else:
				self.expansion[k] += p.expansion[k]

	def simplify(self):
		ind = []
		for k in self.expansion:
			if self.expansion[k] == 0:
				ind.append(k)
		for k in ind:
			self.expansion.pop(k)

	def copy(self):
		p = WalshExpansion(self.n)
		p.expansion = self.expansion.copy()
		return p

	def export(self):
		res = ""

		for k, v in self.expansion.items():
			res += '{"w":' + str(v) + ',"ids":['
			if len(k) > 0:
				for i in k[:-1]:
					res += str(i) + ',' 
				res += str(k[-1])
			res += "]},"

		return res[:-1]


	# x \in {-1, 1}^n
	def eval(self, x):
		res = 0
		#print( self.expansion.items())
		for k, v in self.expansion.items():
			# I'm not an expert in python, maybe there is a more efficient way, sorry...
			parity = True
			for i in k:
				if x[i] == -1:
					parity = not parity

			if parity:
				res += v
			else:
				res -= v

		return res

	# Transform the function by f(xshift xor x)
	# xshift \in {-1, 1}^n
	def xor(self, xshift):
		for k, v in self.expansion.items():
			parity = True
			for i in k:
				if xshift[i] == 1:
					parity =  not parity

			if not parity:
				self.expansion[k] = -v

