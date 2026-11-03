def main():
    import numpy as np
    import bruce
    import argparse
    import matplotlib.pyplot as plt

    from astropy.time import Time, TimeDelta
    from astropy.table import Table
    from astropy import units as u
    from astropy.coordinates import (
        SkyCoord, EarthLocation, AltAz, get_sun, get_body
    )
    from astroplan import Observer
    from astroplan.utils import time_grid_from_range
    from astroquery.mast import Catalogs




    # ------------------------------------------------
    # Argument parsing
    # ------------------------------------------------
    def parse_args():
        parser = argparse.ArgumentParser("spocsearch")

        parser.add_argument("tic_id", type=int)
        parser.add_argument("--t_zero", required=True, type=float)
        parser.add_argument("--period", required=True, type=float)
        parser.add_argument("--width", required=True, type=float)
        parser.add_argument("--start", required=False, type=str, default=None)
        parser.add_argument("--end", required=False, type=str, default=None)
        parser.add_argument("--observatory", required=False, type=str, default='Paranal')

        return parser.parse_args()


    def transit_plan(
        start,
        end,
        resolution,
        tic_id,
        observatory="Paranal",
        sun_max_alt=-15.0,
        target_min_alt=30.0,
        moon_min_seperation=15.0,
    ):
        import numpy as np
        from astropy.time import Time
        from astropy import units as u
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
        from astroplan import Observer
        from astroquery.mast import Catalogs
        from astropy.table import Table

        # ----------------------------------------
        # Target
        # ----------------------------------------
        tic = Catalogs.query_object(f"TIC{tic_id}", radius=0.02, catalog="TIC")[0]
        target = SkyCoord(tic["ra"] * u.deg, tic["dec"] * u.deg)

        # ----------------------------------------
        # Observer
        # ----------------------------------------
        location = EarthLocation.of_site(observatory)
        observer = Observer(location=location, name=observatory)

        # ----------------------------------------
        # Transit cycles within window
        # ----------------------------------------
        first_cycle = int(np.floor((start.jd - args.t_zero) / args.period))
        last_cycle  = int(np.ceil((end.jd - args.t_zero) / args.period))

        rows = []

        # ----------------------------------------
        # Loop over cycles
        # ----------------------------------------
        for cycle in range(first_cycle, last_cycle + 1):
            t_center = args.t_zero + cycle * args.period
            t_start  = t_center - args.width / 2.0
            t_end    = t_center + args.width / 2.0

            if t_end < start.jd or t_start > end.jd:
                continue

            # ----------------------------------------
            # Sample only the transit window
            # ----------------------------------------
            transit_times = Time(
                np.arange(
                    max(t_start, start.jd),
                    min(t_end, end.jd),
                    resolution.to(u.day).value,
                ),
                format="jd",
                scale="utc",
            )

            altaz_tr = AltAz(obstime=transit_times, location=location)
            target_alt = target.transform_to(altaz_tr).alt.deg
            sun_alt = get_sun(transit_times).transform_to(altaz_tr).alt.deg

            visible_mask = (target_alt >= target_min_alt) & (sun_alt <= sun_max_alt)

            if not visible_mask.any():
                continue

            # ----------------------------------------
            # First visible transit time defines night
            # ----------------------------------------
            t_first_visible = transit_times[visible_mask][0]
            from astropy.coordinates import get_sun
            sun = get_sun(t_first_visible)

            sunset  = observer.sun_set_time(t_first_visible, which="previous")
            sunrise = observer.sun_rise_time(t_first_visible, which="next")
            evening_twilight = observer.target_set_time(
                sunset,
                sun,
                which="next",
                horizon=sun_max_alt*u.degree,
                n_grid_points=150
            )

            morning_twilight = observer.target_rise_time(
                sunrise,
                sun,
                which="previous",
                horizon=sun_max_alt*u.degree,
                n_grid_points=150
            )
            # Night grid for visibility calculations
            night_grid = sunset + np.arange(
                -0.2,
                (sunrise - sunset+0.2).to(u.minute).value,
                resolution.to(u.minute).value,
            ) * u.minute

            altaz_n = AltAz(obstime=night_grid, location=location)
            target_alt_n = target.transform_to(altaz_n).alt.deg
            sun_alt_n    = get_sun(night_grid).transform_to(altaz_n).alt.deg

            night_visible_mask = (target_alt_n >= target_min_alt) & (sun_alt_n <= sun_max_alt)
            visible_for = (night_visible_mask.sum() * resolution).to(u.hour).value


            # ----------------------------------------
            # Visible transit duration
            # ----------------------------------------
            t_vis_start = np.min(transit_times[visible_mask].jd)
            t_vis_end   = np.max(transit_times[visible_mask].jd)
            target_rise = np.min(night_grid[night_visible_mask].jd)
            target_set   = np.max(night_grid[night_visible_mask].jd)

            

            in_transit_for = max(0.0, (t_vis_end - t_vis_start) * 24.0)  # hours
            frac_in_transit = in_transit_for / (args.width * 24.0)

            # ----------------------------------------
            # Moon separation (optional)
            # ----------------------------------------
            if moon_min_seperation is not None:
                overlap_times = transit_times[visible_mask]
                moon_sep = np.mean(get_body("moon", overlap_times, location).separation(target).deg)
            else:
                moon_sep = np.nan

            night_label = sunset.datetime.strftime("%Y-%m-%d")

            rows.append([
                tic_id,
                observatory,
                sunset.jd,
                sunrise.jd,
                evening_twilight.jd,
                morning_twilight.jd,
                target_rise,
                target_set,
                night_label,
                cycle,
                visible_for,
                in_transit_for,
                frac_in_transit,
                t_start,
                t_center,
                t_end,
                moon_sep,
                sun_max_alt,
                target_min_alt,
                moon_min_seperation
            ])

        # ----------------------------------------
        # Output Table
        # ----------------------------------------
        if len(rows)==0 : 
            return         Table(
                                    names=[
                                        "tic_id",
                                        "observatory",
                                        "sunset",
                                        "sunrise",
                                        "evening_twilight",
                                        "morning_twilight",
                                        "target_rise",
                                        "target_set",
                                        "night",
                                        "cycle_number",
                                        "visible_for",
                                        "in_transit_for",
                                        "frac_in_transit",
                                        "transit_start_jd",
                                        "transit_center_jd",
                                        "transit_end_jd",
                                        "moon_mean_sep",
                                        "sun_max_alt",
                                        "target_min_alt",
                                        "moon_min_seperation"
                                    ]
                                ) 
        t = Table(
            rows=rows,
            names=[
                    "tic_id",
                    "observatory",
                    "sunset",
                    "sunrise",
                    "evening_twilight",
                    "morning_twilight",
                    "target_rise",
                    "target_set",
                    "night",
                    "cycle_number",
                    "visible_for",
                    "in_transit_for",
                    "frac_in_transit",
                    "transit_start_jd",
                    "transit_center_jd",
                    "transit_end_jd",
                    "moon_mean_sep",
                    "sun_max_alt",
                    "target_min_alt",
                    "moon_min_sep"
            ]
        )

        # Formatting
        t["visible_for"].format = ".1f"
        t["in_transit_for"].format = ".1f"
        t["frac_in_transit"].format = ".2f"
        t["moon_mean_sep"].format = ".1f"
        t["cycle_number"].format = "d"
        t["tic_id"].format = "d"

        return t



    # ------------------------------------------------
    # Run
    # ------------------------------------------------
    args = parse_args()

    t = transit_plan(
        start=Time.now() if args.start is None else Time(args.start),
        end=Time.now() + TimeDelta(20, format="jd") if args.end is None else Time(args.end),
        resolution=1 * u.minute,
        tic_id=args.tic_id, observatory=args.observatory
    )

    if len(t) > 0:
        print(t)
    else:
        print("No observable transits.")

    start_night = (Time.now() if args.start is None else Time(args.start)).strftime("%Y-%m-%d")
    end_night = (Time.now() + TimeDelta(20, format="jd") if args.end is None else Time(args.end)).strftime("%Y-%m-%d")
    output_file_name = 'TIC-{:}_{:}_{:}_to_{:}.fits'.format(args.tic_id, args.observatory,start_night, end_night)
    t.write(output_file_name, overwrite=True, format='fits')


    def jd_to_utc_str(jd, fmt="%H:%M"):
        """Convert JD to a UTC time string."""
        from astropy.time import Time
        return Time(jd, format="jd", scale="utc").to_datetime().strftime(fmt)


    def time_to_utc_str(t, fmt="%H:%M"):
        """Astropy Time -> UTC string."""
        return t.to_datetime().strftime(fmt)


    def plot_transit_altitude(
        row,
        resolution=5*u.minute,
        target_min_alt=30.0,
    ):
        """
        Plot target altitude through the night.
        - Nighttime shaded dark grey
        - Target track in black (dotted below altitude cut)
        - Transit track in red (dotted below altitude cut)
        - Caps at transit start/end
        - Marker at transit midpoint
        """

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.time import Time
        from astropy import units as u
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        from astroplan import Observer
        from astroquery.mast import Catalogs

        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']

        # ------------------------------------
        # Observer
        # ------------------------------------
        location = EarthLocation.of_site(row["observatory"])
        observer = Observer(location=location, name=row["observatory"])

        # ------------------------------------
        # Target coordinates
        # ------------------------------------
        tic = Catalogs.query_object(
            f"TIC{row['tic_id']}", radius=0.02, catalog="TIC"
        )[0]

        target = SkyCoord(
            tic["ra"] * u.deg,
            tic["dec"] * u.deg,
        )

        # ------------------------------------
        # Transit times
        # ------------------------------------
        t_start = Time(row["transit_start_jd"], format="jd", scale="utc")
        t_mid   = Time(row["transit_center_jd"], format="jd", scale="utc")
        t_end   = Time(row["transit_end_jd"], format="jd", scale="utc")

        # ------------------------------------
        # Night bounds
        # ------------------------------------
        t_center = Time(row["night"] + "T12:00:00")
        sunset  = observer.sun_set_time(t_center, which="next")
        sunrise = observer.sun_rise_time(t_center, which="next")

        # ------------------------------------
        # Time grids
        # ------------------------------------
        night_times = sunset + np.arange(
            0,
            (sunrise - sunset).to(u.minute).value,
            resolution.to(u.minute).value,
        ) * u.minute

        transit_times = t_start + np.arange(
            0,
            (t_end - t_start).to(u.minute).value,
            resolution.to(u.minute).value,
        ) * u.minute

        # ------------------------------------
        # Altitudes
        # ------------------------------------
        altaz_night = AltAz(obstime=night_times, location=location)
        altaz_tr    = AltAz(obstime=transit_times, location=location)

        from astropy.coordinates import get_sun

        sun_alt_tr = (
            get_sun(transit_times)
            .transform_to(AltAz(obstime=transit_times, location=location))
            .alt.deg
        )

        alt_night = target.transform_to(altaz_night).alt.deg
        alt_tr    = target.transform_to(altaz_tr).alt.deg

        # ------------------------------------
        # Masks
        # ------------------------------------
        night_above = alt_night >= target_min_alt
        night_below = ~night_above

        tr_above = alt_tr >= target_min_alt
        tr_below = ~tr_above


        # ------------------------------------
        # Visible portion of transit
        # ------------------------------------
        # Transit times sampled at resolution
        sun_max_alt = -18.0  # astronomical night

        visible_mask = (
            (alt_tr >= target_min_alt)
            & (sun_alt_tr <= sun_max_alt)
        )

        t_vis_start = Time(row["target_rise"], format='jd')
        t_vis_end = Time(row["target_set"],format='jd')
        t_vis_start_str = time_to_utc_str(t_vis_start)
        t_vis_end_str   = time_to_utc_str(t_vis_end)

        # if visible_mask.any():
        #     t_vis_start = transit_times[visible_mask][0]
        #     t_vis_end   = transit_times[visible_mask][-1]

        #     t_vis_start_str = time_to_utc_str(t_vis_start)
        #     t_vis_end_str   = time_to_utc_str(t_vis_end)
        # else:
        #     t_vis_start_str = "—"
        #     t_vis_end_str   = "—"


        # ------------------------------------
        # Plot
        # ------------------------------------
        fig,ax = plt.subplots(1,1, figsize=(10, 5))
        #plt.figure(figsize=(10, 5))

        # --- Night shading
        plt.axvspan(
            sunset.datetime,
            sunrise.datetime,
            color="0.85",
            zorder=0,
            label="Nighttime",
        )

        # --- Target altitude (black)
        plt.plot(
            night_times[night_above].datetime,
            alt_night[night_above],
            color="black",
            lw=2,
            label="Target altitude",
        )

        plt.plot(
            night_times[night_below].datetime,
            alt_night[night_below],
            color="black",
            lw=2,
            ls=":",
        )

        # --- Transit track (red)
        if tr_above.any():
            plt.plot(
                transit_times[tr_above].datetime,
                alt_tr[tr_above],
                color="red",
                lw=2,
                label="In transit",
            )

        if tr_below.any():
            plt.plot(
                transit_times[tr_below].datetime,
                alt_tr[tr_below],
                color="red",
                lw=2,
                ls="dotted",
            )

        # --- Transit start/end caps
        alt_start = target.transform_to(
            AltAz(obstime=t_start, location=location)
        ).alt.deg
        alt_end = target.transform_to(
            AltAz(obstime=t_end, location=location)
        ).alt.deg

        plt.plot(
            [t_start.datetime, t_start.datetime],
            [alt_start - 1.5, alt_start + 1.5],
            color="red",
            lw=2,
        )

        plt.plot(
            [t_end.datetime, t_end.datetime],
            [alt_end - 1.5, alt_end + 1.5],
            color="red",
            lw=2,
        )

        # --- Transit midpoint marker
        alt_mid = target.transform_to(
            AltAz(obstime=t_mid, location=location)
        ).alt.deg

        plt.plot(
            t_mid.datetime,
            alt_mid,
            marker="o",
            color="red",
            markersize=6,
        )

        # --- Minimum altitude
        plt.axhline(
            target_min_alt,
            color="k",
            ls="--",
            lw=1,
            alpha=0.6,
            label=f"Alt = {target_min_alt:.0f}°",
        )

        # ------------------------------------
        # Formatting
        # ------------------------------------
        plt.xlabel("UTC time")
        plt.ylabel("Altitude [deg]")


        # ------------------------------------
        # Title block (left-aligned)
        # ------------------------------------
        sunset_str  = time_to_utc_str(Time(row['sunset'], format='jd'))
        sunrise_str = time_to_utc_str(Time(row['sunrise'], format='jd'))
        evening_twilight_str  = time_to_utc_str(Time(row['evening_twilight'], format='jd'))
        morning_twilight_str  = time_to_utc_str(Time(row['morning_twilight'], format='jd'))


        t_start_str = jd_to_utc_str(Time(row["transit_start_jd"], format='jd'))
        t_mid_str   = jd_to_utc_str(Time(row["transit_center_jd"], format='jd'))
        t_end_str   = jd_to_utc_str(Time(row["transit_end_jd"], format='jd'))

        title_lines = [
            (
                f"TIC {row['tic_id']}  |  "
                f"Night: {row['night']}  |  "
                f"{row['observatory']}"
            ),
            (
                f"Sunset {sunset_str} – Sunrise {sunrise_str} UTC  |  "
                f"Max Sun Alt {row['sun_max_alt']:.1f}° -> Twilights {evening_twilight_str} - {morning_twilight_str} UTC"
            ),
            (
                f"Cycle {row['cycle_number']}  |  "
                f"Visible for {row['visible_for']:.1f} h  |  "
                f"In-transit for {row['in_transit_for']:.1f} h  |  "
                f"{row['frac_in_transit']:.2f}% of the transit"
            ),
            (
                f"Full transit time:        {t_start_str} – {t_end_str} UTC "
                f"(mid {t_mid_str} UTC) | Moon sep: {row['moon_mean_sep']:.1f}°"
            ),
            (
                f"Visible time on night:    {t_vis_start_str} – {t_vis_end_str} UTC    (twilight, target alt, sun alt)"
            ),
            (
                f"Transit Visible on night: {time_to_utc_str(t_vis_start) if t_vis_start.jd >= row['transit_start_jd'] else time_to_utc_str(Time(row['transit_start_jd'], format='jd'))} - "
                f"{time_to_utc_str(Time(row['transit_end_jd'], format='jd')) if row['transit_end_jd'] <= t_vis_end.jd else time_to_utc_str(t_vis_end)} UTC    (twilight, target alt, sun alt, transit)"
            ),
        ]

        plt.title(
            "\n".join(title_lines),
            loc="left",
            fontsize=11,
        )



        plt.ylim(0, 90)
        plt.legend(loc="best")
        plt.tight_layout()
        return fig, ax



    for i in range(len(t)):
        fig, _ = plot_transit_altitude(t[i])
        fig.savefig('TIC-{:}_{:}_{:}'.format(args.tic_id, args.observatory, t['night'][i]))
        plt.close()
