cd 

/usr/bin/pto_gen -o project_from_gen.pto 

/usr/bin/cpfind --multirow --celeste -o project_from_gen_and_cpfind.pto project_from_gen.pto

#/usr/bin/celeste_standalone  -o project_from_gen_and_cpfind_and_clean1.pto -i project_from_gen_and_cpfind.pto

#/usr/bin/cpclean -o project_from_gen_and_cpfind_and_clean2.pto project_from_gen_and_cpfind_and_clean1.pto
/usr/bin/cpclean -o project_from_gen_and_cpfind_and_clean2.pto project_from_gen_and_cpfind.pto

/usr/bin/linefind -o project_from_gen_and_cpfind_and_clean2.pto project_from_gen_and_cpfind_and_clean2.pto 

/usr/bin/autooptimiser -a -l -s -m -o project_from_gen_and_cpfind_and_clean2_opt.pto project_from_gen_and_cpfind_and_clean2.pto

#/usr/bin/pano_modify --canvas=AUTO --crop=AUTO -o project_from_gen_and_cpfind_and_clean2_opt_modify.pto project_from_gen_and_cpfind_and_clean2_opt.pto
/usr/bin/pano_modify --center --straighten --canvas=AUTO --crop=AUTO -o project_from_gen_and_cpfind_and_clean2_opt_modify.pto project_from_gen_and_cpfind_and_clean2_opt.pto

#/usr/bin/nona.exe -o out -m TIFF_m project_from_gen_and_cpfind_and_clean2_opt_modify.pto 
#/usr/bin/enblend.exe -o finished.tif out0000.tif out0001.tif out0002.tif out0003.tif out0004.tif out0005.tif out0006.tif out0007.tif

/usr/bin/hugin_executor --stitching --prefix=pano project_from_gen_and_cpfind_and_clean2_opt_modify.pto
