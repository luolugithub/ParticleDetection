from os.path import exists
import pandas as pd

ideal_file = '../../calibration_summary/best_ideal_methods.csv'
output_file = '/home/ncedilni/Documents/rapport juin/methods.csv'
crossval_file = '../../calibration_summary/crossval.csv'


with open(ideal_file) as ideal_fh:
    with open(crossval_file, 'w') as crossval_fh:
        for line in ideal_fh:
            break
        crossval_fh.write('preprocessing,data_binding,centile,'
                          'minimal_shape_surface,minimal_surface_3d,'
                          'minimal_z_thickness,minimal_z_overlap,tp,'
                          'ideal score image wise,fp,image,ex,fn,dd,prec,sens,'
                          'score_ideal_particle_wise,cross validated score,'
                          'pseudo cross validated score,spade ideal threshold,'
                          'method_ideal_idx\n')
        for i, line in enumerate(ideal_fh):
            method_crossval_file = '../../calibration_summary/{}.csv'.format(i)
            if exists(method_crossval_file):
                with open(method_crossval_file) as method_crossval_fh:
                    crossval_values = method_crossval_fh.read()
                crossval_fh.write('{},{},{}\n'.format(line[:-1],
                                                      crossval_values, i))

df = pd.read_csv(crossval_file, index_col=False)
df.sort_values(by='cross validated score', ascending=False, inplace=True)
df.to_csv(crossval_file, index=False)

dico = [('_', ' '),
        ('maximum intensity projection', 'MIP'),
        ('mean difference', 'Différence de moyenne'),
        ('simple thresholding', 'Simple seuillage'),
        ('normalize mean', 'Normalisation de la moyenne'),
        ('normalize variance', 'Normalisation de la variance'),
        ('normalize by median', 'Normalisation par la médiane'),
        ('pseudo bhattacharyya distance', 'Pseudo-coefficient de '
                                         'Bhattacharryya'),
        ('bhattacharyya distance', 'Coefficient de Bhattacharryya'),
        ('MIP ', 'MIP - '),
        ('Normalisation de la moyenne ', 'Normalisation de la moyenne - '),
        ('max min difference', 'Difference <<~max min~>>'),
        ('median difference', 'Difference de médiane'),
        ('t test', 'Test t de Welch'),
        ('do nothing', 'Aucun'),
        ('median filter', 'Filtre médian'),
        ('Normalisation de la variance ', 'Normalisation de la variance - ')]

seen = []

with open(crossval_file) as input_fh:
    with open(output_file, 'w') as output_fh:
        for line in input_fh:
            method = tuple(line.split(',')[:2])
            if method not in seen:
                corrected_line = line
                for traduction in dico:
                    corrected_line = corrected_line.replace(*traduction)
                output_fh.write(corrected_line)
                seen += [method]
