const demoConfig = {
  "samples": {
    "lens_001": {
        "name": "Sample 1",
        "description": "Nearly complete Einstein ring with strong symmetry (\u03b8_E \u2248 1.2\").",
        "metrics": {
            "bicubic": {
                "psnr": 9.643121793030062,
                "ssim": 0.5932372000348598
            },
            "sr_baseline": {
                "psnr": 11.293351948179664,
                "ssim": 0.6267538782564633
            },
            "sr_hybrid": {
                "psnr": 11.050082119641317,
                "ssim": 0.63487463408647
            }
        }
    },
    "lens_002": {
        "name": "Sample 2",
        "description": "Partial arc configuration with visible source structure.",
        "metrics": {
            "bicubic": {
                "psnr": 8.601615373358268,
                "ssim": 0.597982133059861
            },
            "sr_baseline": {
                "psnr": 10.101006679901602,
                "ssim": 0.6635748022208245
            },
            "sr_hybrid": {
                "psnr": 9.92433236150428,
                "ssim": 0.6699115279862629
            }
        }
    },
    "lens_003": {
        "name": "Sample 3",
        "description": "Quad-like lens showing multiply imaged source components.",
        "metrics": {
            "bicubic": {
                "psnr": 14.165123001199616,
                "ssim": 0.7166010486083847
            },
            "sr_baseline": {
                "psnr": 13.360610044638685,
                "ssim": 0.7029658391735074
            },
            "sr_hybrid": {
                "psnr": 13.757645097701367,
                "ssim": 0.7157872016857268
            }
        }
    },
    "lens_004": {
        "name": "Sample 4",
        "description": "Faint Einstein ring obscured by higher noise levels.",
        "metrics": {
            "bicubic": {
                "psnr": 11.280647582750214,
                "ssim": 0.6425762421340215
            },
            "sr_baseline": {
                "psnr": 12.920632054803026,
                "ssim": 0.6691855955425999
            },
            "sr_hybrid": {
                "psnr": 12.825128265421558,
                "ssim": 0.6793346095001868
            }
        }
    },
    "lens_005": {
        "name": "Sample 5",
        "description": "Complex substructure visible in the lensed arc.",
        "metrics": {
            "bicubic": {
                "psnr": 10.464549768369531,
                "ssim": 0.6123785759802957
            },
            "sr_baseline": {
                "psnr": 12.089868695613193,
                "ssim": 0.6417928428591786
            },
            "sr_hybrid": {
                "psnr": 11.976800628121918,
                "ssim": 0.6558377575417195
            }
        }
    },
    "lens_006": {
        "name": "Sample 6",
        "description": "High-magnification fold caustic configuration.",
        "metrics": {
            "bicubic": {
                "psnr": 14.152896893796317,
                "ssim": 0.6292086535495541
            },
            "sr_baseline": {
                "psnr": 14.629294868752929,
                "ssim": 0.6185023308633057
            },
            "sr_hybrid": {
                "psnr": 14.590900327278897,
                "ssim": 0.6316609618239609
            }
        }
    },
    "lens_007": {
        "name": "Sample 7",
        "description": "Small Einstein radius system, barely resolved in LR.",
        "metrics": {
            "bicubic": {
                "psnr": 14.420839045758937,
                "ssim": 0.6423325089297028
            },
            "sr_baseline": {
                "psnr": 14.579760161787163,
                "ssim": 0.6280145289356445
            },
            "sr_hybrid": {
                "psnr": 14.654953521994447,
                "ssim": 0.6435178963115326
            }
        }
    },
    "lens_008": {
        "name": "Sample 8",
        "description": "Bright active galactic nucleus (AGN) host galaxy lens.",
        "metrics": {
            "bicubic": {
                "psnr": 8.9344928751631,
                "ssim": 0.5456862337906666
            },
            "sr_baseline": {
                "psnr": 10.568055214101609,
                "ssim": 0.6381429497254447
            },
            "sr_hybrid": {
                "psnr": 10.568186136770208,
                "ssim": 0.6477093301994746
            }
        }
    },
    "lens_009": {
        "name": "Sample 9",
        "description": "Asymmetric arc due to elliptical lens mass potential.",
        "metrics": {
            "bicubic": {
                "psnr": 8.706367753571158,
                "ssim": 0.6163295601317703
            },
            "sr_baseline": {
                "psnr": 10.027691859445211,
                "ssim": 0.6959220174983516
            },
            "sr_hybrid": {
                "psnr": 9.854707953872875,
                "ssim": 0.7003489230014993
            }
        }
    },
    "lens_010": {
        "name": "Sample 10",
        "description": "Double ring structure suggesting dual source planes.",
        "metrics": {
            "bicubic": {
                "psnr": 11.346390720157665,
                "ssim": 0.6016358369640236
            },
            "sr_baseline": {
                "psnr": 12.616726525008499,
                "ssim": 0.6542214732905064
            },
            "sr_hybrid": {
                "psnr": 12.563328354780548,
                "ssim": 0.663248182215709
            }
        }
    },
    "lens_011": {
        "name": "Sample 11",
        "description": "Gravitationally sheared background galaxy.",
        "metrics": {
            "bicubic": {
                "psnr": 10.25791459717751,
                "ssim": 0.5962084482404443
            },
            "sr_baseline": {
                "psnr": 12.54736323645857,
                "ssim": 0.6626527234092443
            },
            "sr_hybrid": {
                "psnr": 12.560106653671752,
                "ssim": 0.6782430906200895
            }
        }
    },
    "lens_012": {
        "name": "Sample 12",
        "description": "Edge-on spiral source galaxy strongly distorted.",
        "metrics": {
            "bicubic": {
                "psnr": 12.183741838322064,
                "ssim": 0.6201433189740397
            },
            "sr_baseline": {
                "psnr": 13.75710206753952,
                "ssim": 0.6565200052114467
            },
            "sr_hybrid": {
                "psnr": 13.739392661819195,
                "ssim": 0.6709154552285085
            }
        }
    }
},
  "stats": {
    "psnr": {
        "bicubic": 11.179808436887873,
        "sr_baseline": 12.374288613019138,
        "sr_hybrid": 12.33879700688153
    },
    "ssim": {
        "bicubic": 0.6178599800331354,
        "sr_baseline": 0.6548540822488764,
        "sr_hybrid": 0.665949130850095
    }
}
};