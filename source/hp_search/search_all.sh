sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfox --outcome OS --data_type comb"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfox --outcome OS --data_type comb"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfox --outcome OS --data_type mut"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfox --outcome OS --data_type mut"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfox --outcome OS --data_type cna"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfox --outcome OS --data_type cna"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfox --outcome OS --data_type clin"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfox --outcome OS --data_type clin"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfiri --outcome OS --data_type comb"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfiri --outcome OS --data_type comb"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfiri --outcome OS --data_type mut"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfiri --outcome OS --data_type mut"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfiri --outcome OS --data_type cna"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfiri --outcome OS --data_type cna"
sbatch -p general -N 1 -n 1 -c 16 --mem=12g -t 168:00:00 --mail-type=end --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_rf_cv.py --drug folfiri --outcome OS --data_type clin"
sbatch -N 1 -n 1 -p gpu --mem=12g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 search_xgb_cv.py --drug folfiri --outcome OS --data_type clin"
