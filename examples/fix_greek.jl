function main()
    path = "examples/badger_fit_siler_fixed_hmc_5000.jl"
    s = read(path, String)

    s = replace(s, "??_g" => "lambda_g")

    s = replace(s, "??, ??, ?? = model.thetas[j], model.rhos[j], model.phis[j]" =>
                   "theta, rho, phi = model.thetas[j], model.rhos[j], model.phis[j]")

    s = replace(s, "w[1] *= x == 1 ? (1 - ??) : ??" =>
                   "w[1] *= x == 1 ? (1 - phi) : phi")

    s = replace(s, "w[2] *= x == 1 ? (?? * ??) : (1 - ?? * ??)" =>
                   "w[2] *= x == 1 ? (theta * rho) : (1 - theta * rho)")

    s = replace(s, "w[3] *= x == 1 ? ?? : (1 - ??)" =>
                   "w[3] *= x == 1 ? theta : (1 - theta)")

    s = replace(s, "            ?? = model.phis[j]" =>
                   "            phi = model.phis[j]")

    s = replace(s, "            w *= x == 1 ? (1 - ??) : ??" =>
                   "            w *= x == 1 ? (1 - phi) : phi")

    s = replace(s, "            ???? = model.thetas[j] * model.rhos[j]" =>
                   "            theta_rho = model.thetas[j] * model.rhos[j]")

    s = replace(s, "            w *= x == 1 ? ???? : (1 - ????)" =>
                   "            w *= x == 1 ? theta_rho : (1 - theta_rho)")

    s = replace(s, "            ?? = model.thetas[j]" =>
                   "            theta = model.thetas[j]")

    s = replace(s, "            w *= x == 1 ? ?? : (1 - ??)" =>
                   "            w *= x == 1 ? theta : (1 - theta)")

    write(path, s)
end
main()
