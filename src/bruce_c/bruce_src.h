double bruce_loglike(const double y, const double yerr, const double model, const double jitter, const int offset);
double _delta(const double th, const double sin2i, const double omrad, const double ecc);
double __delta (const double th, const double * z0);
double vsign(double a, double b);
double merge(double a , double b, int mask);
double brent_minimum_delta(double ax, double bx, double cx, const double * z0); 
double t_ecl_to_peri(const double t_zero, const double P, const double incl, const double ecc, const double omrad);
double getEccentricAnomaly(double M, const double ecc);
double getTrueAnomaly(const double time, const double e, const double w, const double period,  const double t_zero, const double incl, const int accurate_tp);
double get_z(const double nu, const double e, const double incl, const double w, const double radius_1);
double getProjectedPosition(const double nu, const double w, const double incl);

double clip(double a, double b, double c);
double q1(double z, double p, double c, double a, double g, double I_0);
double q2(double z, double p, double c, double a, double g, double I_0, double eps);
double flux_drop_analytical_power_2(double d_radius, double k, double c, double a, double eps);
double flux_drop_analytical_uniform(double d_radius, double k, double SBR);
double get_intensity_from_limb_darkening_law (int ld_law, double * ldc, double mu_i, int offset);
double area(double z, double r1, double r2);
double flux_drop_annulus(double d_radius, double k, double SBR, int ld_law, double * ldc, int n_annulus, int primary, int offset);
double __lc (const double time,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp);
double _lc (const double time,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp);
double __rv1(const double time,
            const double t_zero, const double period,
            const double K1,
            const double e, const double w, const double incl,
            const double V0,
            const int accurate_tp);
void __rv2(const double time, double * RV1, double * RV2,
            const double t_zero, const double period,
            const double K1, const double K2,
            const double e, const double w, const double incl,
            const double V0,
            const int accurate_tp);
double transit_width(double radius_1, double k, double b, double period);


void linspace(double start, double end, int num, double *array);
void bin_data_fast(
    const double *time, const double *flux, int time_size,
    const double *edges, double *binned_flux, double *binned_flux_err, int binned_size,
    int *count
);
void median_filter_fast(const double *x, const double *y, double *output_signal,
                   double window, int signal_length);
void convolve_1d_fast(const double *x, const double *y, double *output_signal, 
                 double kernel_length, int signal_length);