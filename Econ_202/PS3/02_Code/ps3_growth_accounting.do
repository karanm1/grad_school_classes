/************************
Author: Karan Makakr
Date: Sept 2025
Description: Econ 202 PS3 Growth Accounting
*************************/

clear
set more off
cd "C:\Users\karan\Box\Class Materials\Fall 2025\Econ 202\Problem Sets"


/************************
1. Setup
*************************/

*****************
* Import, Subset
*****************

import excel using "pwt1001.xlsx", sheet("Data") clear firstrow

keep if inlist(countrycode, "USA", "FRA", "KOR", "SGP", "GBR", "DEU")

keep country countrycode year rgdpna emp rnna avh hc labsh

/************************
2. Clean
*************************/


* Define Required Vars
gen total_labor = emp * avh
gen y_l_ratio = rgdpna / (total_labor)
gen k_y_ratio = rnna/rgdpna
rename hc h_l_ratio
bysort countrycode: egen mean_labsh = mean(labsh)
gen mean_alpha_ratio = (1-mean_labsh)/mean_labsh
gen alpha_ratio = (1-labsh)/labsh

* Log Transform
gen ln_y_l_ratio = ln(y_l_ratio)
gen ln_k_y_ratio = ln(k_y_ratio)
gen ln_h_l_ratio = ln(h_l_ratio)

* Reshape to wide
keep countrycode year ln_y_l_ratio ln_k_y_ratio ln_h_l_ratio alpha_ratio mean_alpha_ratio
rename (ln_y_l_ratio ln_k_y_ratio ln_h_l_ratio alpha_ratio) (ln_y_l_ratio_ ln_k_y_ratio_ ln_h_l_ratio_ alpha_ratio_)
reshape wide ln_* alpha_ratio_, i(countrycode) j(year)


/************************
2. Analysis
*************************/

* Calculate growth rates
foreach var in ln_y_l_ratio ln_k_y_ratio ln_h_l_ratio {
    foreach t in 1950_2019 1950_1973 1973_2000 1973_2019 2000_2019 1960_2019 1960_1997 1997_2019 {
        local t1 = substr("`t'", 1, 4)
        local t2 = substr("`t'", 6, 4)
        local T = `t2' - `t1'

        * K/Y Ratio Diffs
        if "`var'" == "ln_k_y_ratio" {
            * Log Diff
            gen d_`var'_`t1'_`t2' = `var'_`t2'*alpha_ratio_`t2' - `var'_`t1'*alpha_ratio_`t1'
            * Annualize
            replace d_`var'_`t1'_`t2' = (d_`var'_`t1'_`t2' / `T')*100

            * Log Diff
            gen d1_`var'_`t1'_`t2' = `var'_`t2'*mean_alpha_ratio - `var'_`t1'*mean_alpha_ratio
            * Annualize
            replace d1_`var'_`t1'_`t2' = (d1_`var'_`t1'_`t2' / `T')*100
        }

        * Other Diffs
        else {
            * Log Diff
            gen d_`var'_`t1'_`t2' = `var'_`t2' - `var'_`t1'
            * Annualize
            replace d_`var'_`t1'_`t2' = (d_`var'_`t1'_`t2' / `T')*100
        }
    }
}

keep countrycode d_* d1_*


* Calculate TFP Growth

foreach t in 1950_2019 1950_1973 1973_2000 1973_2019 2000_2019 1960_2019 1960_1997 1997_2019 {
    * Allow alpha to vary
    gen d_ln_A_`t' = d_ln_y_l_ratio_`t' - d_ln_k_y_ratio_`t' - d_ln_h_l_ratio_`t'

    * Hold alpha constant
    gen d1_ln_A_`t' = d_ln_y_l_ratio_`t' - d1_ln_k_y_ratio_`t' - d_ln_h_l_ratio_`t'
}


**********************
* Print Results
**********************

* USA, FRA, GBR, DEU
foreach c in USA FRA GBR DEU {
    foreach t in 1950_2019 1950_1973 1973_2000 1973_2019 2000_2019 {

        preserve
        keep if countrycode == "`c'"
        di "`c', `t'"
        di "Y/L Growth: " d_ln_y_l_ratio_`t'[1]
        di "K/Y Growth: " d_ln_k_y_ratio_`t'[1]
        di "K/Y Growth (Constant Alpha): " d1_ln_k_y_ratio_`t'[1]
        di "H/L Growth: " d_ln_h_l_ratio_`t'[1]
        di "TFP Growth: " d_ln_A_`t'[1]
        di "TFP Growth (Constant Alpha): " d1_ln_A_`t'[1]
        restore
    }
}

* KOR, SGP
foreach c in KOR SGP {
    foreach t in 1960_2019 1960_1997 1997_2019 {
        preserve
        keep if countrycode == "`c'"
        di "`c', `t'"
        di "Y/L Growth: " d_ln_y_l_ratio_`t'[1]
        di "K/Y Growth: " d_ln_k_y_ratio_`t'[1]
        di "K/Y Growth (Constant Alpha): " d1_ln_k_y_ratio_`t'[1]
        di "H/L Growth: " d_ln_h_l_ratio_`t'[1]
        di "TFP Growth: " d_ln_A_`t'[1]
        di "TFP Growth (Constant Alpha): " d1_ln_A_`t'[1]
        restore
    }
}


**********************
* Make Tables
**********************

gen gr_1950_2019 =.
gen gr_1950_1973 =.
gen gr_1973_2000 =.
gen gr_1973_2019 =.
gen gr_2000_2019 =.

label variable gr_1950_2019 "1950-2019"
label variable gr_1950_1973 "1950-1973"
label variable gr_1973_2000 "1973-2000"
label variable gr_1973_2019 "1973-2019"
label variable gr_2000_2019 "2000-2019"

**** Western Countries
local i = 1
foreach c in USA FRA GBR DEU {
    foreach x in d_ln_y_l_ratio d_ln_h_l_ratio d_ln_k_y_ratio d_ln_A d1_ln_k_y_ratio d1_ln_A {
        foreach t in 1950_2019 1950_1973 1973_2000 1973_2019 2000_2019 {
            replace gr_`t' = `x'_`t' if countrycode == "`c'"
        }
        eststo t`i': estpost summarize gr_* if countrycode == "`c'"
        local i = `i' + 1
    }
}

  esttab t1 t2 t3 t4 t5 t6  using growth_accounting_west.tex, ///
  cells("mean(pattern(1 1 1 1 1 1) fmt(a2))") ///
  replace label nonum noobs ///
  mtitle("Y/L" "H/L" "K/Y" "TFP" "K/Y" "TFP Growth") /// 
  mgroups("""Variable $\alpha$" "Constant $\alpha$", pattern(1 0 1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) ///
  prehead(`"\begin{tabular}{lccccccc}"' `"\hline \hline"') ///
  posthead(`"& (1) & (2) & (3) & (4) & (5) & (6) \\"' `"\hline \\ "' `"\textit{Panel A: USA} \\"') ///
  prefoot(`"\\ "') ///
  postfoot(`"\hline \\ "') ///
  collabels(none)

  esttab t7 t8 t9 t10 t11 t12  using growth_accounting_west.tex, ///
  cells("mean(pattern(1 1 1 1 1 1) fmt(a2))") ///
  append label ///
  nomtitle nonum noobs /// 
  prehead(`""') ///
  posthead(`"\textit{Panel B: France} \\"') ///
  prefoot(`"\\ "') ///
  postfoot(`"\hline \\ "') ///
  collabels(none)

  esttab t13 t14 t15 t16 t17 t18  using growth_accounting_west.tex, ///
  cells("mean(pattern(1 1 1 1 1 1) fmt(a2))") ///
  append label ///
  nomtitle nonum noobs /// 
  prehead(`""') ///
  posthead(`"\textit{Panel C: UK} \\"') ///
  prefoot(`"\\ "') ///
  postfoot(`"\hline \\ "') ///
  collabels(none)

  esttab t19 t20 t21 t22 t23 t24  using growth_accounting_west.tex, ///
  cells("mean(pattern(1 1 1 1 1 1) fmt(a2))") ///
  append label ///
  nomtitle nonum noobs /// 
  prehead(`""') ///
  posthead(`"\textit{Panel D: Germany} \\"') ///
  prefoot(`"\\ "') ///
  postfoot(`"\hline \\ "' `"\end{tabular}"') ///
  collabels(none)


**** Korean and Singapore

gen gr_1960_2019 =.
gen gr_1960_1997 =.
gen gr_1997_2019 =.

label variable gr_1960_2019 "1960-2019"
label variable gr_1960_1997 "1960-1997"
label variable gr_1997_2019 "1997-2019"

local i = 1
foreach c in KOR SGP {
    foreach x in d_ln_y_l_ratio d_ln_h_l_ratio d_ln_k_y_ratio d_ln_A d1_ln_k_y_ratio d1_ln_A {
        foreach t in 1960_2019 1960_1997 1997_2019 {
            replace gr_`t' = `x'_`t' if countrycode == "`c'"
        }
        eststo t`i': estpost summarize gr_1960_2019 gr_1960_1997 gr_1997_2019 if countrycode == "`c'"
        local i = `i' + 1
    }
}

  esttab t1 t2 t3 t4 t5 t6  using growth_accounting_east.tex, ///
  cells("mean(pattern(1 1 1 1 1 1) fmt(a2))") ///
  replace label nonum noobs ///
  mtitle("Y/L" "H/L" "K/Y" "TFP" "K/Y" "TFP Growth") /// 
  mgroups("""Variable $\alpha$" "Constant $\alpha$", pattern(1 0 1 0 1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) ///
  prehead(`"\begin{tabular}{lccccccc}"' `"\hline \hline"') ///
  posthead(`"& (1) & (2) & (3) & (4) & (5) & (6) \\"' `"\hline \\ "' `"\textit{Panel A: Korea} \\"') ///
  prefoot(`"\\ "') ///
  postfoot(`"\hline \\ "') ///
  collabels(none)

  esttab t7 t8 t9 t10 t11 t12  using growth_accounting_east.tex, ///
  cells("mean(pattern(1 1 1 1 1 1) fmt(a2))") ///
  append label ///
  nomtitle nonum noobs /// 
  prehead(`""') ///
  posthead(`"\textit{Panel B: Singapore} \\"') ///
  prefoot(`"\\ "') ///
  postfoot(`"\hline \\ "' `"\end{tabular}"') ///
  collabels(none)