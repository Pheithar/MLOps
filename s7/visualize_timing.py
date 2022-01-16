import matplotlib.pyplot as plt

num_workers = list(range(8, 17))

avg = [6.086470444997151,
       5.808094342549642,
       5.834463357925415,
       5.711818218231201,
       5.686939875284831,
       5.7366251945495605,
       5.7594083944956465,
       5.77534556388855,
       5.780792713165283
       ]

std = [0.17439520466090386,
       0.047725158477872974,
       0.027918896469354207,
       0.021832296809766605,
       0.03729569312718301,
       0.04452274466314854,
       0.0827539024300348,
       0.0637799285561891,
       0.022737499997609142
       ]

plt.errorbar(num_workers, avg, yerr=std)
plt.show()