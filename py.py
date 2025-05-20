import statistics

a = [0.9654, 0.9599, 0.9280,
0.9602,
0.9695,
0.9777,
0.9855,
0.9576,
0.9284,
0.9697,
0.9383,
0.9160
]
dc = [0.9663, 0.9726, 0.9625, 0.9243, 0.8968, 0.9267, 0.9619, 0.9383, 0.9481, 0.9488, 0.9507, 0.9411]
print(statistics.median(a), statistics.mean(a))
print(statistics.median(dc), statistics.mean(dc))