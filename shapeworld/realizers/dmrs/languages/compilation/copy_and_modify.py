import os
import sys


skip = 0
count = 0
with open(os.path.join(sys.argv[1], 'lexicon.tdl'), 'r') as filehandle1:
    with open(os.path.join(sys.argv[2], 'lexicon.tdl'), 'w') as filehandle2:
        for line in filehandle1:
            if skip > 0:
                skip -= 1
            elif line == 'less_than_deg := av_-_dg_le &\n':
                filehandle2.write('less_than_deg := av_-_dg-jomv_le &\n')
                count += 1
            elif line == 'more_than_deg := av_-_dg_le &\n':
                filehandle2.write('more_than_deg := av_-_dg-jomv_le &\n')
                count += 1
            elif line == 'quarter_n1 := n_pp_c-of_le &\n':
                filehandle2.write('quarter_n1 := n_pp_c-gr-of_le &\n')
                count += 1
            elif line == 'a_couple_of_adj := aj_-_i-num_le &\n':
                skip = 4
                count += 1
            elif line == 'a_couple_a_adj := aj_-_i-num_le &\n':
                skip = 4
                count += 1
            elif line == 'blue_a1 := aj_-_i-color-er_le &\n':
                skip = 4
                count += 1
            elif line == 'circle_n1 := n_pp_c-of_le &\n':
                skip = 4
                count += 1
            elif line == 'cross_n1 := n_-_c_le &\n':
                skip = 4
                count += 1
            elif line == 'cyan_a1 := aj_-_i-color_le &\n':
                skip = 4
                count += 1
            # elif line == 'either_conj := c_xp_either-mrk_le &\n':
            #     skip = 3
            #     count += 1
            elif line == 'further_a1 := aj_pp-pp_i-cmp_le &\n':
                skip = 4
                count += 1
            elif line == 'gray_a1 := aj_-_i-color-er_le &\n':
                skip = 4
                count += 1
            elif line == 'green_a1 := aj_-_i-color-er_le &\n':
                skip = 4
                count += 1
            elif line == 'magenta_a1 := aj_-_i-color_le &\n':
                skip = 4
                count += 1
            elif line == 'pentagon_n1 := n_-_c_le &\n':
                skip = 4
                count += 1
            elif line == 'rectangle_n1 := n_-_c_le &\n':
                skip = 4
                count += 1
            elif line == 'red_a1 := aj_-_i-color-er_le &\n':
                skip = 4
                count += 1
            elif line == 'round_n1 := n_pp_c-of_le &\n':
                skip = 4
                count += 1
            elif line == 'shape_n1 := n_-_mc-ed_le &\n':
                skip = 4
                count += 1
            elif line == 'square_n1 := n_pp_c-of_le &\n':
                skip = 4
                count += 1
            elif line == 'square_n2 := n_-_c-meas_le &\n':
                skip = 4
                count += 1
            elif line == 'square_v1 := v_np_le &\n':
                skip = 4
                count += 1
            elif line == 'triangle_n1 := n_-_c_le &\n':
                skip = 4
                count += 1
            elif line == 'upper_a1 := aj_-_i_le &\n':
                skip = 4
                count += 1
            elif line == 'yellow_a1 := aj_-_i-color-er_le &\n':
                skip = 4
                count += 1
            # elif line == 'whether_conj := c_xp_whether-mrk_le &\n':
            #     skip = 3
            #     count += 1
            else:
                filehandle2.write(line)
assert count == 24, count

skip = 0
count = 0
with open(os.path.join(sys.argv[1], 'mtr.tdl'), 'r') as filehandle1:
    with open(os.path.join(sys.argv[2], 'mtr.tdl'), 'w') as filehandle2:
        for line in filehandle1:
            if skip > 0:
                skip -= 1
            elif line == 'be_cop_prd_1x_rule := arg0e+1x_gtr &\n':
                filehandle2.write(line)
                filehandle2.write(' [ CONTEXT [ RELS <! [ PRED "~_[paj](?:_|$)",\n')
                filehandle2.write('                       ARG0 [ E [ MOOD indicative ] ] ] !> ] ].\n')
                skip = 5
                count += 1
            else:
                filehandle2.write(line)
assert count == 1, count


skip = 0
count = 0
with open(os.path.join(sys.argv[1], 'trigger.mtr'), 'r') as filehandle1:
    with open(os.path.join(sys.argv[2], 'trigger.mtr'), 'w') as filehandle2:
        for line in filehandle1:
            if skip > 0:
                skip -= 1
            elif line == 'be_c_are_prd_1x_rule := be_cop_prd_1x_rule &\n':
                filehandle2.write(line)
                filehandle2.write(' [ CONTEXT [ RELS <! [ ARG0 [ E [ TENSE present ] ] ] !> ],\n')
                filehandle2.write('   FLAGS.TRIGGER "be_c_are" ].\n')
                skip = 3
                count += 1
            else:
                filehandle2.write(line)
assert count == 1, count
