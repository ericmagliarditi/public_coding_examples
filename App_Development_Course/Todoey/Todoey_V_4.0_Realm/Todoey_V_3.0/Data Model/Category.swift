//
//  Category.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/24/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import Foundation
import RealmSwift

class Category: Object {
    @objc dynamic var name: String = ""
    /**
     We now want to develop the parent/child relationship of the data
     Do that in the following manner:
     create an items array using the List object - A realm object
     NEED TO DECLARE THE VARIABLE TYPE IN THE List Object
     You initialize the list as an empty array of sorts
     This allows us to define the forward relationship
     Category --> Items
    */
    let items = List<Item>()
    @objc dynamic var colorCat: String = ""
}
