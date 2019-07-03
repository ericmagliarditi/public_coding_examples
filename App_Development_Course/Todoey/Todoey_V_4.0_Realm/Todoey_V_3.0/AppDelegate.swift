//
//  AppDelegate.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/17/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import UIKit
import CoreData
import RealmSwift

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        
        //Lets see where realm is - it is a realm file
        //Need to download piece of software to view
//        print(Realm.Configuration.defaultConfiguration.fileURL)
        
        //Add it to database
        /**
         do this by using realm.write
        */
        do {
            _ = try Realm()
        }
        catch{
            print("Error generating the realm object \(error)")
        }
        
        
        
        
        
        
        return true
    }


}

