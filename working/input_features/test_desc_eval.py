import descriptor_eval as de
for a in [1.2, 1.5, 1.8]:
    print('Alpha: ', a)
    de.evaluate_descriptors(dataname='zhu_rat',
                            descriptors_name='RDKitDescriptors',
                            target='LD50',
                            alpha=a)

