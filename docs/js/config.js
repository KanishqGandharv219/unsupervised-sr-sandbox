const demoConfig = {
    "samples": {
        "lens_001": {
            "name": "Sample 1",
            "description": "Nearly complete Einstein ring with strong symmetry (\u03b8_E \u2248 1.2\").",
            "metrics": {
                "bicubic": {
                    "psnr": 14.26781993399821,
                    "ssim": 0.905741461037739
                },
                "sr_baseline": {
                    "psnr": 14.854376724383641,
                    "ssim": 0.8916335209725728
                },
                "sr_hybrid": {
                    "psnr": 15.013692184565695,
                    "ssim": 0.8901038418981323
                }
            }
        },
        "lens_002": {
            "name": "Sample 2",
            "description": "Partial arc configuration with visible source structure.",
            "metrics": {
                "bicubic": {
                    "psnr": 15.725063519328842,
                    "ssim": 0.8845187953089704
                },
                "sr_baseline": {
                    "psnr": 15.633777735173954,
                    "ssim": 0.8575536253043715
                },
                "sr_hybrid": {
                    "psnr": 15.824284416510341,
                    "ssim": 0.8548851595675477
                }
            }
        },
        "lens_003": {
            "name": "Sample 3",
            "description": "Quad-like lens showing multiply imaged source components.",
            "metrics": {
                "bicubic": {
                    "psnr": 14.758089053888185,
                    "ssim": 0.9259996639739603
                },
                "sr_baseline": {
                    "psnr": 16.68076991971661,
                    "ssim": 0.9181158855362539
                },
                "sr_hybrid": {
                    "psnr": 16.82239926447162,
                    "ssim": 0.9179017630612808
                }
            }
        },
        "lens_004": {
            "name": "Sample 4",
            "description": "Faint Einstein ring obscured by higher noise levels.",
            "metrics": {
                "bicubic": {
                    "psnr": 13.65100604640193,
                    "ssim": 0.934450285418422
                },
                "sr_baseline": {
                    "psnr": 12.652072098784071,
                    "ssim": 0.9347789250004164
                },
                "sr_hybrid": {
                    "psnr": 13.218695834748404,
                    "ssim": 0.9401718038759957
                }
            }
        },
        "lens_005": {
            "name": "Sample 5",
            "description": "Complex substructure visible in the lensed arc.",
            "metrics": {
                "bicubic": {
                    "psnr": 15.403912672942784,
                    "ssim": 0.9404917452550813
                },
                "sr_baseline": {
                    "psnr": 16.054162222410984,
                    "ssim": 0.9287566209874171
                },
                "sr_hybrid": {
                    "psnr": 16.035970276141764,
                    "ssim": 0.922821817797798
                }
            }
        },
        "lens_006": {
            "name": "Sample 6",
            "description": "High-magnification fold caustic configuration.",
            "metrics": {
                "bicubic": {
                    "psnr": 15.760922846439216,
                    "ssim": 0.907119457489245
                },
                "sr_baseline": {
                    "psnr": 16.742308150897824,
                    "ssim": 0.8892470627810527
                },
                "sr_hybrid": {
                    "psnr": 16.610676405869828,
                    "ssim": 0.8842845121476086
                }
            }
        },
        "lens_007": {
            "name": "Sample 7",
            "description": "Small Einstein radius system, barely resolved in LR.",
            "metrics": {
                "bicubic": {
                    "psnr": 14.597888429601166,
                    "ssim": 0.917806149556719
                },
                "sr_baseline": {
                    "psnr": 16.45533221397271,
                    "ssim": 0.9111196052872185
                },
                "sr_hybrid": {
                    "psnr": 16.460899248500745,
                    "ssim": 0.90703702986453
                }
            }
        },
        "lens_008": {
            "name": "Sample 8",
            "description": "Bright active galactic nucleus (AGN) host galaxy lens.",
            "metrics": {
                "bicubic": {
                    "psnr": 16.046229630751,
                    "ssim": 0.9126459209693598
                },
                "sr_baseline": {
                    "psnr": 16.33818876130127,
                    "ssim": 0.8932214709226923
                },
                "sr_hybrid": {
                    "psnr": 16.236294287567276,
                    "ssim": 0.8881864010330897
                }
            }
        },
        "lens_009": {
            "name": "Sample 9",
            "description": "Asymmetric arc due to elliptical lens mass potential.",
            "metrics": {
                "bicubic": {
                    "psnr": 15.273749365708504,
                    "ssim": 0.9271900608834415
                },
                "sr_baseline": {
                    "psnr": 16.829975876914588,
                    "ssim": 0.9159653534964347
                },
                "sr_hybrid": {
                    "psnr": 16.593655856788757,
                    "ssim": 0.9116572631029113
                }
            }
        },
        "lens_010": {
            "name": "Sample 10",
            "description": "Double ring structure suggesting dual source planes.",
            "metrics": {
                "bicubic": {
                    "psnr": 15.93836670048825,
                    "ssim": 0.9223294946654089
                },
                "sr_baseline": {
                    "psnr": 16.727802584964046,
                    "ssim": 0.9075686325995732
                },
                "sr_hybrid": {
                    "psnr": 16.416224786219438,
                    "ssim": 0.9002798843185343
                }
            }
        },
        "lens_011": {
            "name": "Sample 11",
            "description": "Gravitationally sheared background galaxy.",
            "metrics": {
                "bicubic": {
                    "psnr": 14.984691545252943,
                    "ssim": 0.9494850074387334
                },
                "sr_baseline": {
                    "psnr": 16.02248346115118,
                    "ssim": 0.9435527009820824
                },
                "sr_hybrid": {
                    "psnr": 16.02232257779121,
                    "ssim": 0.9403135526690557
                }
            }
        },
        "lens_012": {
            "name": "Sample 12",
            "description": "Edge-on spiral source galaxy strongly distorted.",
            "metrics": {
                "bicubic": {
                    "psnr": 14.2377031417487,
                    "ssim": 0.9408083099530795
                },
                "sr_baseline": {
                    "psnr": 12.995557644188944,
                    "ssim": 0.9370620134114935
                },
                "sr_hybrid": {
                    "psnr": 13.515474998686363,
                    "ssim": 0.9384740212584506
                }
            }
        }
    },
    "stats": {
        "arc_sharpness": {
            "bicubic": 221.96,
            "sr_baseline": 268.50,
            "sr_hybrid": 267.41,
            "sr_unsupervised": 249.40,
            "hr_ground_truth": 266.39
        },
        "ring_contrast": {
            "bicubic": 5.08,
            "sr_baseline": 4.97,
            "sr_hybrid": 5.39,
            "sr_unsupervised": 4.83,
            "hr_ground_truth": 5.14
        }
    }
};